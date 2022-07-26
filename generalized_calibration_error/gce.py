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

from sys import float_repr_style
from .components.distance_fns import tvd
from .components.aggregation_fns import expectation
from .components.estimation_schemes.histogram_binning_estimators import uniform_histogram_binning_estimator

class gce:
    """Generalized Calibration Errror

    Allows for modular construction of calibration error metrics as defined by the paper "What is Your Metric Telling You? 
    Evaluating Classifier Calibration under Context-Specific Definitions of Reliability".

    Each component is used as follows, starting from two (n x c) arrays of row vectors, the first being classifier output vectors 
    (`probs`), and the second being labels (`labels`).  Note: in all cases where two arrays are taken as arguments or returned,
    the first of the two are output vectors and the second are label vectors.

    If `preselection` is `True`, then `selection_op` is applied.  The selection operation should be defined to take two (n x c) 
    numpy arrays and return a tuple of two (n' x c) numpy arrays, where n' <= n.

    Then the `lens` function is applied. The lens function should be defined to take two (n x c) numpy arrays and return
    two (n x c') vectors.

    If `preselection` is `False`, then the `selection_op` is applied here instead of before the lens function is applied.

    Finally, the `esimtation_scheme` is used to compute error. The `estimation_scheme` should extend the BaseEstimator class,
    and uses both the `distance_fn`, and `aggregation_fn` to compute error.  The `distance_fn` should be defined to take 
    two c-dimensional numpy arrays and return a float.  The `aggregation_fn` should be defined to take two (b x c) numpy arrays
    , and a distance_fn, and return a float.

    Examples of each of these components can be found in the `components` module.

    Attributes:
        lens (func): Lensing function that induces a classification problem. Default is a no op ("full" lens)
        distance_fn (func): Distance function for error calculation.  Default is total variation distance. 
        aggregation_fn (func): Function to aggregate errors over disparate distance function calculations.  Default is 
            expectation.
        selection_op (func): Function that selects instances in which to compute error.  Default is a no op (all
            instances are selected)
        preselection (bool): Whether to apply the selection operator before (True) or after (False) applying the lensing 
            function.  Default is False.
        estimation_scheme (components.estimation_schemes.BaseEstimator): Class to estimate the error defined by a lens,  
            distance function, and aggregation scheme.  Default is uniform bin histogram binning with 15 bins.
    """
    def __init__(self, 
                 lens              = lambda x,y: (x,y),
                 distance_fn       = tvd,
                 aggregation_fn    = expectation,
                 selection_op      = lambda x,y: (x,y), 
                 preselection      = False,
                 estimation_scheme = uniform_histogram_binning_estimator(bins = 15)) -> None:
        self.lens              = lens
        self.selection_op      = selection_op
        self.estimation_scheme = estimation_scheme
        self.preselection      = preselection

        self.estimation_scheme.set_distance_fn(distance_fn)
        self.estimation_scheme.set_aggregation_fn(aggregation_fn)                         

    def __call__(self, probs, labels) -> float:
        """Computes the calibration error between `probs` and `labels`

        Args:
            probs (numpy.array): An (n x c) row vector corresponding to n probabilistic classifier outputs over c classes.
            labels (numpy.array): An (n x c) row vector corresponding to n one hot labels over c classes.  

        Returns:
            float : Calibration error between `probs` and `labels`
        """
        # "Preselection" means selection operators before lensing
        if self.preselection:
            probs, labels = self.selection_op(probs, labels) # Note: selection operators happen AFTER lensing
        
        # Induce a new classification problem via lensing function
        probs, labels = self.lens(probs,labels)
        
        # Not "preselection" (postselection) means selection operators after lensing
        if not self.preselection:
            probs, labels = self.selection_op(probs, labels) # Note: selection operators happen AFTER lensing

        # Return CE based on estimation scheme
        return self.estimation_scheme(probs, labels)


