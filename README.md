# MVPA MEG Tutorial

[work in progress]

This tutorial accompanies the preprint titled "Multivariate pattern analysis for MEG: a comprehensive comparison of dissimilarity measures", which is available at [placeholder](http://doi.org/).

## Python tutorial

### Preparation
This tutorial can be downloaded as a zip file from [placeholder](https://github.com/m-guggenmos/megmvpa/tutorial_python.zip). In addition to [Jupyter-Notebook](https://jupyter.org/)-based tuturials, this zip file includes the example dataset used for this tutorial. To reduce computational costs, the dataset is for one participant only and includes only 10 of 92 experimental conditions.

Content of the zip file:

File | Description
--- | --- 
cv.py | _containing code for pseudo-trials/permutations/cross-validation_
dissimilarity.py | _containing a number of custom dissimilarity measures_
weird.py | _weighted robust distance classifier (WeiRD), see also [here](https://github.com/m-guggenmos/weird)_
data01_sess1_bs10.pkl | _data for subject 1, session 1_
data01_sess2_bs10.pkl | _data for subject 1, session 2_
labels01_sess1.pkl | _trial labels for subject 1, session 1_
labels01_sess2.pkl | _trial labels for subject 1, session 2_

In addition, the tutorial requires 4 established scientific python packages: numpy, scipy, scikit-learn, matplotlib

### List of tutorials:
* [Decoding](https://github.com/m-guggenmos/megmvpa/blob/master/python_decoding.ipynb)
* [RDMs and Reliability](https://github.com/m-guggenmos/megmvpa/blob/master/python_reliability.ipynb)
* [Distance measures and cross-validation](https://github.com/m-guggenmos/megmvpa/blob/master/python_distance.ipynb)

## Matlab tutorial (in preparation)
* [placeholder](https://github.com/m-guggenmos/megmvpa/blob/master/matlab_decoding.ipynb)
