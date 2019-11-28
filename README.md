# Outlines
- Implementation of MDLP (http://yaroslavvb.com/papers/fayyad-discretization.pdf)

The paper suggests one of discretization methods (broader concept of binning).
1. Its purpose is to get cutpoints which converts numerical columns into categorical ones.
2. To get the cutpoints, its hypothesis assumes `Occam's razor`, which states that selecting simpler, shorter hypothesis(`length`) is desirable.
3. This leads to minimizes statistics calculated from a numerical column: `length(P(H)) + length(P(H|T))`.
- `length` is measured by `Entropy`
- and it also uses target information

You can refer to below formulas for more details.
![formula_1](/assets/5.png)
![formula_2](/assets/4.png)
![formula_3](/assets/3.png)
![formula_4](/assets/2.png)
![formula_5](/assets/1.png)

# Features
- Can be executed as `multiprocess` by `n_jobs`
- Works as `sklearn` way

# How to run
```python
from discretization.mdlp import *
mdlp = MDLP(con_features=data.feature_names, base=2, max_cutpoints=2, n_jobs=-1)
mdlp.fit_transform(X)
```

# Random thoughts on the paper
Pros = Discretization is done related with target information which leads to performance  
Cons = If features are correlated, discretized features can be redundant (Needs of `feature selection`)

# Results
= You can check the results in the notebook files. Model performances are improved in most cases.

# References
- Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning (http://yaroslavvb.com/papers/fayyad-discretization.pdf)
- Discretization: An Enabling Technique (https://cs.nju.edu.cn/zhouzh/zhouzh.files/course/dm/reading/reading03/liu_dmkd02.pdf)
- Minimum Description Length Principle (https://www.cs.helsinki.fi/u/ttonteri/pub/roosmdlencyc2016.pdf)

