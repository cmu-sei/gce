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

from gettext import install
from setuptools import setup, find_packages
import pathlib
import os

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
	name = 'generalized_calibration_error',
	version = '0.1',
	description = 'Generalized Calibration Error',
	long_description = long_description,
	long_description_content_type = 'text/markdown',
	url = 'https://github.com/cmu-sei/gce',
	author = 'Eric Heim, John Kirchenbauer, Jacob Oaks',
	author_email = 'etheim@sei.cmu.edu',
	classifiers = [
		'Programming Language :: Python :: 3',
		'Development Status :: 4 - Beta',
		'Intended Audience :: Developers',
		'Topic :: Scientific/Engineering',
		'License :: MIT (SEI)'
	],
	packages=[
		'generalized_calibration_error', 
		'generalized_calibration_error.components',
		'generalized_calibration_error.components.estimation_schemes',
	],
	python_requires = '>=3.6, <4'
	)