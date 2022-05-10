# Identification of main content in the works included in the HathiTrust Extracted Features dataset
Code for clustering digitized pages of the works based on the features that are available through the [HathiTrust Extracted Features dataset v.2.0](https://wiki.htrc.illinois.edu/pages/viewpage.action?pageId=79069329) with the aim of separating main content of a work from paratextual elements.
Reference: A. Lucic, R. Burke and J. Shanahan, "Unsupervised Clustering with Smoothing for Detecting Paratext Boundaries in Scanned Documents,"
2019 ACM/IEEE Joint Conference on Digital Libraries (JCDL), 2019, pp. 53-56, doi: 10.1109/JCDL.2019.00018. The conference paper is available [here](https://ieeexplore.ieee.org/abstract/document/8791148)
## Running the code
Several python libraries are required for running the code and they are included in the requirements.txt file. The code depends on the methods developed under the htrc-feature-reader python library. This library can be installed using either pip or conda package manager:
[pip install htrc-feature-reader](https://pypi.org/project/htrc-feature-reader/)
[conda install -c htrc htrc-feature-reader](https://anaconda.org/htrc/htrc-feature-reader)
## Motivation for the development of this method
This work started as part of the [Reading Chicago Reading project](https://dh.depaul.press/reading-chicago/) at DePaul University in 2018. Reading Chicago Reading project received the HathiTrust Research Center Advanced Collaborative computational support grant to explore a set of in copyright and out of copyright works related to the analysis of the [One Book One Chicago](https://www.chipublib.org/one-book-one-chicago/) program that were included in the [Extracted Features](https://analytics.hathitrust.org/datasets) dataset. To be able to limit the extraction of features to main content we needed to establish where the main content begins and ends. If the elements of the book such as Table of Contents, Epilogue, Bibliography, Critical Introduction are not excluded before extracting text measures from non-fiction and fiction works, these elements can skew the metrics (e.g. count of locations or personal names in the work). Paratext boundaries are not a consistent metadata element that accompany digital files included in digital libraries. Even if such information exists in the accompanying metadata files, this information needs to be verified.
## Modeling paratext as the outlier of main work
The conclusion of the work was that paratext elements lend themselves to being modeled as outliers of main work. As the amount of paratext increases in a volume, however, it is harder to establish the beginning and end of the main content. 
## Future work
We plan to continue developing this method to establish the upper bounds of accuracy with which paratext elements can be identified and excluded from digital files. We also plan to explore the degree to which different paratext elements lends themselves to being identified in a work using automated methods.
