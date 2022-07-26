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

import numpy as np
from typing import Tuple

class label_selection_op:
    """Label selection operator
    
    Selects instances with specified class labels

    Attributes:
        classes (list(int)) : Class label indices to be selected.
    """
    def __init__(self, classes) -> None:
        self.classes = classes

    def __call__(self, probs, labels) -> Tuple[np.array, np.array]:
        """ Applies the label selection operator to ``probs`` and ``labels``

        Returns all rows of ``probs`` and ``labels`` where there is a 1 a column with index
        specified in ``classes`` (i.e. selects all instances with label in ``classes``)

        Args:
            probs (np.array) : An (n x c) matrix of row vectors, each of which is a probabilistic
                output of a classifier.
            labels (np.array) : An (n x c) matrix of row vectors, each of which is a one hot vector
                label for a c-way classification problem.

        Returns:
            tuple[np.array. np.array] : ``probs`` and ``labels`` after the label selection operator 
            is applied.  Each of which is now a (m x c) matrix, where m <= n.
        """
        # Assumes labels are correct one hot vectors (each row will have exactly one nonzero entry)
        numerical_labels = np.nonzero(labels)[1]
        selection_idxs = [i for i in range(numerical_labels.shape[0]) if numerical_labels[i] in self.classes]
        return probs[selection_idxs,:], labels[selection_idxs,:]

class output_selection_op:
    """Output selection operator
    
    Selects instances based on condition on probabilistic classifier outputs

    Attributes:
        operator (operator) : Logical operator to compare to probabilistic outputs
        rhs_value (float) : Value to compare to probabilistic outputs
        membership_dim (int): Index of the probablistic output vector to compare
    """
    def __init__(self, operator, rhs_value, membership_dim=1) -> None:
        # TODO:  Need a check here to make sure it's a logical operator
        self.operator = operator
        self.rhsvalue = rhs_value
        self.membership_dim = membership_dim

    def __call__(self, probs, labels) -> Tuple[np.array, np.array]:
        """ Applies the output selection operator to ``probs`` and ``labels``

        Returns all rows of ``probs`` and ``labels`` that satisfies the condition defined by
        ``probs``, ``operator``, ``rhs_value``, and ``membership_dim``.  For instance, if 
        ``probs = [[0.1, 0.9],  [0.3, 0.7]]``, ``operator = operator.gt``, ``rhs_value = 0.8``,
        and ``membership_dim = 1``, the first instance (row) would be selection because 0.9 > 0.8.
        The second instance would be selected because !(0.7 > 0.8). 

        Args:
            probs (np.array) : An (n x c) matrix of row vectors, each of which is a probabilistic
                output of a classifier.
            labels (np.array) : An (n x c) matrix of row vectors, each of which is a one hot vector
                label for a c-way classification problem.

        Returns:
            tuple[np.array. np.array] : ``probs`` and ``labels`` after the output selection operator 
            is applied.  Each of which is now a (m x c) matrix, where m <= n.
        """
        idxs = np.where(self.operator(probs[:,self.membership_dim], self.rhsvalue))[0].tolist()
        return probs[idxs,:], labels[idxs,:]