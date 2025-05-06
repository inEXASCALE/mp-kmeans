# mpkmeans


A mixed-precision algorithm of $k$-means is designed towards understanding of the low precision arithmetic for Euclidean distance computations and analyzing the issues using low precision arithmetic for unnormalized data. 

By performing simulations across data with various settings, we showcase that decreased precision for $k$-means computing only results in a minor increase in sum of squared errors while not necessarily leading to degrading performance regarding clustering results. The robustness of the mixed-precision $k$-means algorithms over various precisions is demonstrated. Fully reproducible experimental code is included in this repository, which illustrates the potential application of using mixed-precision k-means over various data science tasks including data clustering and image segmentation.

The dependencies for running our code and data loading:

- [classixclustering](https://github.com/nla-group/classix). (For preprocessed UCI data loading)
- NumPy (The fundamental package for scientific computing)
- Pandas (For data format and storage)
- scikit-learn (Machine Learning in Python)
- opencv-python (For image segmentation)
- pychop (For low precision arithmetic simulation)

Details on the underlying algorithms can be found in the technical report:

One can install them before running our code via:
```Bash
pip install classixclustering torch tqdm scikit-learn opencv-python
```

We also requires the installation of ``pychop`` of version 0.3.0, to install, use:
```Bash
 pip install pychop==0.3.0 
```


The repository contains the folder:

- ``data``: data used for the simulations
- ``results``: experimental results (figures and tables)
- ``src``: simulation code of mixed-precision k-means and distance computing

This repository contains the following algorithms for k-means computing:
* StandardKMeans1  - the native kmeans algorithm using distance (4.3)
* StandardKMeans2 - the native kmeans algorithm using distance (4.4)  
* mpKMeans - the mixed-precision kmeans algorithm using Algorithm 6.3
* allowKMeans1 - kmeans performed in full low precision for computing distance (4.3)
* allowKMeans2 - kmeans performed in full low precision for computing using distance (4.4)

One can load the library via 

```Python
from src.kmeans import <classname> # e.g., from src.kmeans import mpKMeans
```

The following example showcases the useage of ``mpkmeans`` class

```Python
from pychop import chop
from src.kmeans import mpKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import adjusted_rand_score # For clustering quality evaluation

X, y = make_blobs(n_samples=2000, n_features=2, centers=5) # Generate data with 5 clusters

LOW_PREC = chop(prec='q52') # Define quarter precision
mpkmeans = mpKMeans(n_clusters=5, seeding='d2', low_prec=LOW_PREC, random_state=0, verbose=1)
mpkmeans.fit(x)

print(adjusted_rand_score(y, mpkmeans.labels)) # load clustering membership via mpkmeans.labels
```

Note that for half and single preicison simulation, user can directly use the built-class in our software via:

```Python
from src.kmeans import chop
import numpy as np

LOW_PREC = chop(np.float16)
```


All empirical results in paper can be produced via the bash command in Linux:
```Python
python3 run_all.py
```

After code running is completed, one can find the results in the folder ``results``.


References
------------

```
C., Erin, X. Chen, X. Liu. “Computing $k$-means in mixed precision.” (2024).
```

The bibtex:
```bibtex
@misc{carson2024,
     title={Computing $k$-means in mixed precision}, 
     author={Erin Carson and Xinye Chen and Xiaobo Liu},
     year={2024},
     eprint={2407.12208},
     archivePrefix={arXiv},
     primaryClass={math.NA},
     url={https://arxiv.org/abs/2407.12208}, 
}
```
