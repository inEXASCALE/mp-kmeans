# mpkmeans


A mixed-precision algorithm of $k$-means is designed towards understanding of the low precision arithmetic for Euclidean distance computations and analyzing the issues using low precision arithmetic for unnormalized data. 

By performing simulations across data with various settings, we showcase that decreased precision for $k$-means computing only results in a minor increase in sum of squared errors while not necessarily leading to degrading performance regarding clustering results. The robustness of the mixed-precision $k$-means algorithms over various precisions is demonstrated. Fully reproducible experimental code is included in this repository, which illustrates the potential application of using mixed-precision k-means over various data science tasks including data clustering and image segmentation.

The dependencies for running our code and data loading:

- [classixclustering](https://github.com/nla-group/classix). (For preprocessed UCI data loading)
- NumPy (The fundamental package for scientific computing)
- Pandas (For data format and storage)

Details on the underlying algorithms can be found in the technical report:

Note that for half and single preicison simulation, user can directly use the built-class in our software via:




References
------------

```
E. Carson, X. Chen, and X. Liu. Computing k-means in mixed precision. ArXiv:2407.12208 [math.NA], July 2024.
```

The bibtex:
```bibtex
@techreport{ccl24,
  author = "Erin Carson and Xinye Chen and Xiaobo Liu",
  title = "Computing $k$-means in Mixed Precision",
  month = jul,
  year = 2024,
  type = "{ArXiv}:2407.12208 [math.{NA}]",
  url = "https://arxiv.org/abs/2407.12208"
}
```
