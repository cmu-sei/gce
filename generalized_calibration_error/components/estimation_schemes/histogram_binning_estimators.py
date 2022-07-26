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

"""
Note:  The uniform and adaptive binning schemes are reliant on the package 
https://github.com/uu-sml/calibration
"""

from abc import ABC, abstractproperty
import abc

from ..aggregation_fns import expectation
from ..distance_fns import tvd
from .base_estimator import BaseEstimator

# The "calibration" package that contains histogram estimation implementations is here: https://github.com/uu-sml/calibration
from calibration.binning.uniform import UniformBinning
from calibration.binning.datadependent import DataDependentBinning
from calibration.binning.general import BinningTree

# TODO: Special case of num_bins = 1 not handled
class BaseHistogramBinningEstimator(BaseEstimator, ABC):
    """Abstract base class for histogram binning estimators
    
    Defines the basic algorithm agnostic of the binning strategy employed

    Attributes:
        binning (class) : Abstract binning strategt to be employed by the binning tree algorithm
    """
    @abstractproperty
    def binning(self):
        pass

    def __init__(self) -> None:
        self.binning_tree = BinningTree(self.binning)
        super().__init__()
    
    def __call__(self, probs, labels) -> float:
        """ Bins instances and then finds the aggregate calibration error from mean bin values
        
        The algorithms proceeds as follows, first the ``binning`` strategy is used to bin each row
        of ``probs`` and ``labels`` according to ``probs``.  Then the mean label and prob for each
        bin is found.  Then error is found between the mean label and prob in each bin using 
        ``distance_fn``.  Finally, the errors for all bins are aggregated to a single scalar using
        ``aggregation_fn``.

        Args:
            probs (np.array) : An (n x c) matrix of row vectors, each of which is a probabilistic
                output of a classifier.
            labels (np.array) : An (n x c) matrix of row vectors, each of which is a one hot vector
                label for a c-way classification problem.

        Returns:
            float : The calibration error between ``probs`` and ``labels``.
        """
        self.binning_tree.fit(probs)
        binned_probs = self.binning_tree.bin_data()
        binned_labels = None if labels is None else self.binning_tree.bin_data(labels)
        return self.aggregation_fn(binned_probs, binned_labels, self.distance_fn)


class uniform_histogram_binning_estimator(BaseHistogramBinningEstimator):
    """Uniform histogram binning estimator
    
    Defines a histogram estimator of calibration error where the binning strategy is to define
    a number of bins of equal size, uniformly distributed over the probability simplex.

    This scheme is used in `Guo, Chuan, et al. "On calibration of modern neural networks." International 
    Conference on Machine Learning. PMLR, 2017. <http://proceedings.mlr.press/v70/guo17a/guo17a.pdf>`_

    Attributes:
        bins (int) : The number of bins in which to partition the probability simplex
    """
    binning = None
    def __init__(self, bins=15) -> None:
        self.binning = UniformBinning(bins=bins)
        super().__init__()

# Weird case:  Because we set the min_number of entries per bin based on a given fraction of the number of instances give, we can't
# make the super classes init make the binning tree in this init, as we don't have the instances until the call of the estimator.
# This shouldn't incur unneccesary additional computation overhead though as the heavy lifting is in fit.
class adaptive_histogram_binning_estimator(BaseHistogramBinningEstimator):
    """Adaptive histogram binning estimator
    
    Defines a histogram estimator of calibration error where the binning strategy is dependent on
    the distribution of ``probs``.  Here, bin bounds are determined by approximately achieving
    ``frac_per_bin`` instances per bin using a bottom-up partitioning scheme.

    This scheme is used in `Vaicenavicius, Juozas, et al. "Evaluating model calibration in 
    classification." The 22nd International Conference on Artificial Intelligence and 
    Statistics. PMLR, 2019. 
    <http://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf>`_


    Attributes:
        frac_per_bin (float) : The target proportion of instances per bin in the adaptive binnin
            strategy.
    """
    binning = None
    def __init__(self, frac_per_bin = 0.1) -> None:
        self.frac_per_bin = frac_per_bin

    def __call__(self, probs, labels) -> float:
        N = probs.shape[0]
        min_size = round(N*self.frac_per_bin)
        self.binning = DataDependentBinning(min_size = min_size, threshold = "mean")

        # Now I can call the super's init and call
        super().__init__()
        return super().__call__(probs=probs, labels=labels)