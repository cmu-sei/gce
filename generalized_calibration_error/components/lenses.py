# Generalized Calibration Error

# Copyright 2022 Carnegie Mellon University.

# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE
# MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO
# WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER 
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR 
# MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. 
# CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT
# TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

# Released under a MIT (SEI)-style license, please see license.txt or contact 
# permission@sei.cmu.edu for full terms.

# [DISTRIBUTION STATEMENT A] This material has been approved for public release 
# and unlimited distribution.  Please see Copyright notice for non-US Government 
# use and distribution.

# This Software includes and/or makes use of the following Third-Party Software 
# subject to its own license:

# 1. calibration (https://github.com/uu-sml/calibration/blob/master/LICENSE) 
# Copyright 2019 Carl Andersson, David Widmann.

# 2. NumPy (https://github.com/numpy/numpy/blob/main/LICENSE.txt) 
# Copyright 2005-2022 NumPy Developers.

# DM22-0406

from functools import reduce
from collections import Counter
from typing import Tuple

import numpy as np

class top_k_lens:
    """Top-k lens
    
    Induces a k+1-way classification problem of the k-highest probability classes versus
    all other classes.

    Defined in `Vaicenavicius, Juozas, et al. "Evaluating model calibration in 
    classification." The 22nd International Conference on Artificial Intelligence and 
    Statistics. PMLR, 2019. 
    <http://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf>`_

    Attributes:
        k (int) : Defines the ``k`` most probable classes to be retained in the lens
    """
    def __init__(self, k=5) -> None:
        self.k = k
    
    def __call__(self, probs, labels) -> Tuple[np.array, np.array]:
        """Applies the Top-k lens to ``probs`` and ``labels``

        Args:
            probs (np.array) : An (n x c) matrix of row vectors, each of which is a probabilistic
                output of a classifier.
            labels (np.array) : An (n x c) matrix of row vectors, each of which is a one hot vector
                label for a c-way classification problem.

        Returns:
            tuple[np.array. np.array] : ``probs`` and ``labels`` after the Top-k lens is applied.  
            Each of which is now a (n x k+1) matrix.
        """
        # Number of instances and classes, respectively
        N, C = probs.shape

        # Check if number of classes exceeds the k in top-k
        if self.k > C:
            raise ValueError(f'k value must be <= C, num classes , (got k={self.k}, for C={C})')

        top_idxs = np.argsort(probs, axis=1)[:, -1:-(self.k+1):-1]
        _top_probs = probs[np.repeat(
            np.arange(N).reshape(-1, 1), self.k, axis=1), top_idxs]

        top_probs = np.concatenate(
            [_top_probs, 1 - np.sum(_top_probs, axis=1, keepdims=True)], axis=1)
        
        # build new label representing "label in topk indices"
        _top_labels = labels[np.repeat(np.arange(N).reshape(-1, 1), self.k, axis=1), top_idxs]

        top_labels = np.concatenate(
            [_top_labels, 1 - np.sum(_top_labels, axis=1, keepdims=True)], axis=1)

        return top_probs, top_labels

def top_1_lens(probs, labels):
    """Top-1 lens

        Special case of ``top_k_lens`` with k=1.  Induces a binary classification problem of the
        most probable class versus the rest.

        Used in `Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht. "Obtaining well calibrated 
        probabilities using bayesian binning." Twenty-Ninth AAAI Conference on Artificial Intelligence. 
        2015. <https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9667/9958>`_ for binary
        classification and in `Guo, Chuan, et al. "On calibration of modern neural networks." International 
        Conference on Machine Learning. PMLR, 2017. <http://proceedings.mlr.press/v70/guo17a/guo17a.pdf>`_
        for multi-class classification.

        Args:
            probs (np.array) : An (n x c) matrix of row vectors, each of which is a probabilistic
                output of a classifier.
            labels (np.array) : An (n x c) matrix of row vectors, each of which is a one hot vector
                label for a c-way classification problem.

        Returns:
            tuple[np.array. np.array] : ``probs`` and ``labels`` after the Top-1 lens is applied.  
            Each of which is now a (n x 2) matrix.
    """
    return top_k_lens(k=1)(probs, labels)

class class_marginal_lens:
    """Class-marginal lens
    
    Induces a binary classification problem of the specified class versus the rest.

    Defined in `Nixon, Jeremy, et al. "Measuring Calibration in Deep Learning." CVPR Workshops. Vol. 2.
    No. 7. 2019.
    <https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf>`_
    and `Kumar, Ananya, Percy S. Liang, and Tengyu Ma. "Verified uncertainty calibration." Advances in 
    Neural Information Processing Systems 32 (2019). 
    <https://proceedings.neurips.cc/paper/2019/file/f8c0c968632845cd133308b1a494967f-Paper.pdf>`_


    Attributes:
        class_num (int) : Specifies the class to marginalize versus the others
    """
    def __init__(self, class_num) -> None:
        # Can pass in a group dictionary, but keys will get thrown away
        if type(class_num) is not int:
            raise TypeError(f'"class_num" should be an positive integer identifier of a class')
        
        if class_num < 0:
            raise TypeError(f'"class_num" should be a positive integer identifier of a class')

        self.class_num = class_num

    def __call__(self, probs, labels) -> Tuple[np.array, np.array]:
        """Applies the ``class_num`` class-marginal lens to ``probs`` and ``labels``

        Args:
            probs (np.array) : An (n x c) matrix of row vectors, each of which is a probabilistic
                output of a classifier.
            labels (np.array) : An (n x c) matrix of row vectors, each of which is a one hot vector
                label for a c-way classification problem.

        Returns:
            tuple[np.array. np.array] : ``probs`` and ``labels`` after the ``class_num`` 
            class-marginal lens is applied.  Each of which is now a (n x 2) matrix.
        """
        C = probs.shape[1]

        # Checks if any given class index in groupings exceed number of classes in original classification problem
        if self.class_num > labels.shape[1]:
            raise ValueError(f'"class_num" is larger than the number of classes specified by "labels"')

        marginal_probs = np.vstack((probs[:,self.class_num], 
                                    np.hstack(np.sum(np.delete(probs, self.class_num, axis=1), axis=1)))).T
        marginal_labels = np.vstack((labels[:,self.class_num], 
                                     np.hstack(np.sum(np.delete(labels, self.class_num, axis=1), axis=1)))).T
        return marginal_probs, marginal_labels

class group_lens:
    """Class grouping lens
    
    Induces a classification problem between groups of classes.

    Defined in `Vaicenavicius, Juozas, et al. "Evaluating model calibration in 
    classification." The 22nd International Conference on Artificial Intelligence and 
    Statistics. PMLR, 2019. 
    <http://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf>`_

    Attributes:
        groups (list(list(int))) : List that contains each group in the induced problem. A group is defined
            by a list of the class indices within that group.  For instance, ``[[0,1,2],[3,4]]`` defines
            a binary classification problem between the first three and last two classes.
    """
    def __init__(self, groups) -> None:
        # Can pass in a group dictionary, but keys will get thrown away
        if type(groups) is dict:
            groups = groups.values()

        # Keep track of what classes are used in the grouping dict
        self.used_classes = reduce(lambda a,b: a+b, groups)

        # Check if any of the classes were used more than once (either within a group or between two groups)
        if len([v for v in Counter(self.used_classes).values() if v > 1]) > 0:
            raise ValueError(f'"groups" contains one or more class indices used multiple times (a class can appear only once and in a single group)')

        self.groups = groups

    def __call__(self, probs, labels) -> Tuple[np.array, np.array]:
        """Applies the grouping lens to ``probs`` and ``labels`` with groups specified in ``groups``

        Args:
            probs (np.array) : An (n x c) matrix of row vectors, each of which is a probabilistic
                output of a classifier.
            labels (np.array) : An (n x c) matrix of row vectors, each of which is a one hot vector
                label for a c-way classification problem.

        Returns:
            tuple[np.array. np.array] : ``probs`` and ``labels`` after the grouping lens is applied.
            Each of which is now a (n x \|num_groups\|) matrix.
        """
        C = probs.shape[1]

        # Checks if any given class index in groupings exceed number of classes in original classification problem
        if any([class_label > (C-1) for class_label in self.used_classes]):
            raise ValueError(f'"A class label in "groups" is higher than the number of classes in original classification problem"')

        
        if set(self.used_classes) != set(range(C)):
            raise ValueError(f'"groups" does not cover all classes in original classification problem')

        # Sum the columns defined by the list in each element of the group list and then concatenate the resulting columns
        grouped_probs = np.array([np.sum(probs[:,group], axis=1) for group in self.groups]).T
        grouped_labels = np.array([np.sum(labels[:,group], axis=1) for group in self.groups]).T

        return grouped_probs, grouped_labels