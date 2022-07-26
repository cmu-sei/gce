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

from abc import ABC

class BaseEstimator(ABC):
    """Abstract base estimator class
    
    Provides functions to set required members for use in the ``gce`` class.
    """
    def __init__(self) -> None:
        super().__init__()

    def set_distance_fn(self, distance_fn) -> None:
        """Setter for ``distance_fn``
        """
        self.distance_fn = distance_fn

    def set_aggregation_fn(self, aggregation_fn) -> None:
        """Setter for ``aggregation_fn``
        """
        self.aggregation_fn = aggregation_fn