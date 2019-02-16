# MDLP
1. Implementation of MDLP (http://yaroslavvb.com/papers/fayyad-discretization.pdf)
  - Works `sklearn` way
2. Supervised discretization using target, entropy
3. Can be configured to multiprocess (`n_jobs`)

```python
from discretization.mdlp import *
mdlp = MDLP(con_features=data.feature_names, base=2, max_cutpoints=2, n_jobs=-1)
mdlp.fit_transform(X)
```

# Concept
= Pick hypothesis that minimize length(P(H)) + length(P(H|T))  
= In this paper, length is measured by `Entropy`  
= When it comes to choose cutpoints, pick ones that minimize `Entropy`  

# Pros & Cons
Pros = Discretization is done related with target information  
Cons = If features are correlated, discretized features can be redundant  

TODO  
- dataset to s3
