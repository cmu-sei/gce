from abc import ABC, abstractmethod

import numpy as np

def tvd(prob, label) -> float:
    """Standard total variation distance

    Args:
        prob (np.array) : A c-dimensional probabilistic classifier output.
        label (np.array) : A c-dimensional one-hot label.

    Returns:
        float : The total variation distance between ``prob`` and ``label``
    """
    return np.linalg.norm(np.subtract(prob, label), ord=1, axis=-1) / 2

class generalized_mahalanobis_distance():
    """Generalized Mahalanobis distance.  
    
    Common in metric learning: 
    `Kulis, Brian. "Metric learning: A survey." Foundations and Trends® in Machine 
    Learning 5.4 (2013): 287-364. <https://ieeexplore.ieee.org/abstract/document/8186753>`_

    Attributes:
        W (np.array) : A (c x c) dimensional precision matrix. Default is identity.
    """
    def __init__(self, W=None) -> None:
        self.W = W

    def __call__(self, prob, label) -> float:
        """ Computes the generalized Mahalanobis distance between prob and label
        parameterized by ``W``

        Args:
            prob (np.array) : A c-dimensional probabilistic classifier output.
            label (np.array) : A c-dimensional one-hot label.

        Returns:
            float : The generalized Mahalanobis distance between ``prob`` and ``label``
        """
        # Identity is default, but dimension isn't known until call
        if self.W is None:
            self.W = np.eye(prob.shape[0])
        
        diff = np.subtract(prob, label)
        return np.sqrt(np.dot(np.dot(diff,self.W),np.transpose(diff)))  

class interval_tvd():
    """Interval supressed total variation distance  
    
    Total variation distance conditioned on whether the label is inside of a given interval.
    Note: For this metric, ``label`` only makes sense as aggregate of one-hot labels,
    such as in histogram binning estimation schemes.

    Note 2: Meant to be used in the binary case, though it does not prohibit use in the 
    multi-class case.

    Upper bounds ``inter_interval_distance``.

    Attributes:
        interval (tuple) : A 2-ple of floats containing the lower and upper bound of the interval.
            Default is [0.0, 1.0].  No interval is provided, inclusivity is set to [True, True].
        inclusivity (tuple): A 2-ple of bools indicating whether to include the upper and lower bound
            in the interval check of the labels.
        membership_dim (int): Indicates the index in the label vector corresponding to the class
            in which interval-membership is checked.
    """
    def __init__(self, interval=None, inclusivity=[False,True], membership_dim=1) -> None:
        # If no interval is provided, then the default is going to be all of 
        # [0.0, 1.0], inclusive (identical to normal TVD)
        if interval is None:
            self.interval = [0.0, 1.0]
            self.inclusivity = [True, True]
        else:
            self.interval = interval
            self.inclusivity = inclusivity
        # Note that default is meant to assume the binary case where one-hot, binary classification
        # vectors have the positive class in the second dimension.
        self.membership_dim = membership_dim
    
    def __call__(self, prob, label) -> float:
        """ Computes the total variation distance conditioned on whether the label is inside of 
        ``interval``.

        Returns 0 if ``label`` is inside of ``interval`` in the ``membership_dim`` element.
        Otherwise it returns the tvd between ``prob`` and ``label``

        Args:
            prob (np.array) : A c-dimensional probabilistic classifier output.
            label (np.array) : A c-dimensional one-hot label.

        Returns:
            float : The interval-conditioned total variation distance between ``prob`` and ``label``
        """
        # Check whether label is inside of the lower bound
        if self.inclusivity[0]:
            above_low = label[self.membership_dim]  >= self.interval[0]
        else:
            above_low = label[self.membership_dim]  > self.interval[0]

        # Check whether the label is inside of the upper bound
        if self.inclusivity[1]:
            below_high = label[self.membership_dim]  <= self.interval[1]
        else:
            below_high = label[self.membership_dim]  < self.interval[1]

        if above_low and below_high:
            return 0
        else:
            return tvd(prob, label)
        
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

class inter_interval_distance():
    """Inter-interval distance 
    
    Distance of a label to the closest interval bound, or 0 if within the interval.
    Defined in Equation 7 of "What is Your Metric Telling You? Evaluating Classifier Calibration 
    under Context-Specific Definitions of Reliability"

    Note: For this metric, ``label`` only makes sense as aggregate of one-hot labels,
    such as in histogram binning estimation schemes.

    Note 2: Meant to be used in the binary case, though it does not prohibit use in the 
    multi-class case.

    Lower bounds ``interval_tvd`.

    Attributes:
        interval (tuple) : A 2-ple of floats containing the lower and upper bound of the interval.
        membership_dim (int): Indicates the index in the label vector corresponding to the class
            in which interval-membership is checked.
    """
    def __init__(self, interval=[0.0,1.0], membership_dim=1) -> None:
        # If no interval is provided, then the default is going to be all of [0.0, 1.0], inclusive (identical to normal TVD)
        self.interval = interval
        self.membership_dim = membership_dim
        
    def __call__(self, prob, label) -> float:
        """ Computes inter-interval distance.

        Returns 0 if ``label`` is inside of ``interval`` in the ``membership_dim`` element.
        Otherwise it returns the absolute difference between ``label`` and the closest
        interval boundary.

        Args:
            prob (np.array) : A c-dimensional probabilistic classifier output.
            label (np.array) : A c-dimensional one-hot label.

        Returns:
            float : The interval-conditioned total variation distance between ``prob`` and ``label``
        """
        mean_label_1d = label[self.membership_dim]
        lower_dist = self.interval[0] - mean_label_1d
        upper_dist = mean_label_1d - self.interval[1]
        return max([max([lower_dist, upper_dist]), 0])