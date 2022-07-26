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

def expectation(probs, labels, distance_fn) -> float:
    """Finds the expectation of the distance between ``probs`` and ``labels``

        Args:
            probs (numpy.array): An (b x c) matrix of row vectors corresponding to b probabilistic 
                classifier outputs (or their bin means in the binned case) over c classes.
            labels (numpy.array): An (b x c) matrix of row vectors corresponding to b one hot labels 
                (or their bin means in the binned case) over c classes.  
            distance_fn (func): A distance function that computes the error between a
                probs vector and a labels vector

        Returns:
            float : The expected calibration error over ``probs`` and ``labels`` according to 
            ``distance_fn``    
    """
    proportions = np.array([x.shape[0] for x in labels])
    n = np.sum(proportions)
    proportions = proportions / n

    return np.dot(proportions,np.array([distance_fn(x.mean(axis=0),y.mean(axis=0)) for x, y in zip(probs, labels)]))

def maximum(probs, labels, distance_fn) -> float:
    """Finds the maximum distance between ``probs`` and ``labels`` vectors

        Args:
            probs (numpy.array): An (b x c) matrix of row vectors corresponding to b probabilistic 
                classifier outputs (or their bin means in the binned case) over c classes.
            labels (numpy.array): An (b x c) matrix of row vectors corresponding to b one hot labels 
                (or their bin means in the binned case) over c classes.  
            distance_fn (func): A distance function that computes the error between a
                probs vector and a labels vector

        Returns:
            float : The maximum error between ``probs`` and ``labels`` vectors according to 
            ``distance_fn``    
    """
    return max([distance_fn(x.mean(axis=0),y.mean(axis=0)) for x, y in zip(probs, labels)])