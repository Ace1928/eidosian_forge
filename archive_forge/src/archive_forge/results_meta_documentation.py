import numpy as np
from statsmodels.tools.testing import Holder

example from Kacker 2004, computed with R metafor

> y = c(61.0, 61.4 , 62.21, 62.3 , 62.34, 62.6 , 62.7 , 62.84, 65.9)
> v = c(0.2025, 1.2100, 0.0900, 0.2025, 0.3844, 0.5625, 0.0676, 0.0225, 1.8225)
> res = rma(y, v, data=dat, method="PM", control=list(tol=1e-9))
> convert_items(res, prefix="exk1_metafor.")

