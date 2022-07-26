<!-- Generalized Calibration Error

Copyright 2022 Carnegie Mellon University.

NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE
MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO
WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER 
INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR 
MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. 
CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT
TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

Released under a MIT (SEI)-style license, please see license.txt or contact 
permission@sei.cmu.edu for full terms.

[DISTRIBUTION STATEMENT A] This material has been approved for public release 
and unlimited distribution.  Please see Copyright notice for non-US Government 
use and distribution.

This Software includes and/or makes use of the following Third-Party Software 
subject to its own license:

1. calibration (https://github.com/uu-sml/calibration/blob/master/LICENSE) 
Copyright 2019 Carl Andersson, David Widmann.

2. NumPy (https://github.com/numpy/numpy/blob/main/LICENSE.txt) 
Copyright 2005-2022 NumPy Developers.

DM22-0406 -->

# Generalized Calibration Error Python Package

This package provides the means to create custom calibration metrics from components and then use them to evaluate classification model.

The calibration error framework is outlined in ["What is Your Metric Telling You? Evaluating Classifier Calibration under Context-Specific Definitions of Reliability"](https://arxiv.org/abs/2205.11454).

To get started, take a look at the `GCEExamples.ipynb` notebook, and the `PaperReplication.ipynb` notebook if you're familiar with the paper.  Also consider building the API documentation for easy reference.

## Install Package
To install the package remotely:
```shell
    pip install git+https://github.com/cmu-sei/gce.git
```

After cloning the repo, inside of the base directory, you install the package with:
```shell
    pip install .
```
To also install requirements add `-r requirements.txt` add the end of `pip install .`

## Dependencies
Dependencies can be found in `requirements.txt`.

- [NumPy](https://numpy.org/) - For basic linear algebraic objects and their manipulation.
- [David Widmann's calibration library](https://github.com/uu-sml/calibration/tree/7bd1a2407f96f87e37d81eadaea7efeb14bb8a83) - For histogram binning algorithms.

## Build Documentation (requires [Sphinx](https://www.sphinx-doc.org/en/master/))
To build the API documentation in html form run:
```shell
    cd docs
    make html
```
To view the documentation point your favorite web browser at "docs/build/html/index.html".

## Basic Usage 
The top level class for constructing a calibration metric is gce class:
```python
from generalized_calibration_error import gce
full_ece = gce()
full_ece(probs, labels)
```
Different components can be passed to the object upon construction.  Then, the object can be called with outputs from a probabilistic classifier and one hot labels to compute error using the metric.\

## Defining a Metric with Components 
To define more specific calibration metrics, the gce class can be passed different component arguments.
For instance, to define a Top-1 expected calibration error metric, you can pass the `top_1_lens` function as a `lens` argument:

```python
from generalized_calibration_error import gce
from generalized_calibration_error.components.lenses import top_1_lens 
top1_ece = gce(lens = top_1_lens)
top1_ece(probs, labels)
```
To see what other components can be passed, see the API documentation (or source code) for the `gce` class.  To see what predefined components are provided see the documentation for the `components` module (or the source code for modules therein).  Note that user defined components can be used as long as they are compatible, meaning they take the proper arguments and return the proper objects.  See the different predefined components for examples if you want to write your own components.
