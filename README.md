# MVPA MEG Tutorial

[work in progress]

This tutorial accompanies the preprint titled "Multivariate pattern analysis for MEG: a comprehensive comparison of dissimilarity measures", which is available at [biorxiv](http://www.biorxiv.org/content/early/2017/08/04/172619).

## Python tutorial

### Preparation
This tutorial is based on [IPython/Jupyter Notebook](https://jupyter.org/) files, which are linked below. In addition, the tutorial can be downloaded as a [zip file](https://github.com/m-guggenmos/megmvpa/blob/master/tutorial_python.zip), which includes the notebook files, additional code files and the example dataset used for this tutorial. To reduce computational costs, the dataset is for one participant only and includes only 9 of 92 experimental conditions.

Content of the zip file:

File | Description
--- | --- 
cv.py | _containing code for pseudo-trials/permutations/cross-validation_
dissimilarity.py | _containing a number of custom dissimilarity measures_
weird.py | _weighted robust distance classifier (WeiRD), see also [here](https://github.com/m-guggenmos/weird)_
python_decoding.ipynb | Notebook on _Decoding_
python_reliability.ipynb | Notebook on _RDMs and Reliability_
python_distance.ipynb | Notebook on _Distance measures and cross-validation_
data01_sess1.npy | _data for subject 1, session 1_
data01_sess2.npy | _data for subject 1, session 2_
labels01_sess1.npy | _trial labels for subject 1, session 1_
labels01_sess2.npy | _trial labels for subject 1, session 2_

In addition, the tutorial requires 4 established scientific python packages: numpy, scipy, scikit-learn, matplotlib

### List of tutorials:
* [Decoding](https://github.com/m-guggenmos/megmvpa/blob/master/tutorial_python/python_decoding.ipynb)
* [RDMs and Reliability](https://github.com/m-guggenmos/megmvpa/blob/master/tutorial_python/python_reliability.ipynb)
* [Distance measures and cross-validation](https://github.com/m-guggenmos/megmvpa/blob/master/tutorial_python/python_distance.ipynb)

## Matlab tutorial (in preparation)
* [Decoding](https://github.com/m-guggenmos/megmvpa/blob/master/tutorial_matlab/matlab_decoding.ipynb)
