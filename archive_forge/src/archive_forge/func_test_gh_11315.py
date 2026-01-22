import numpy as np
import scipy.special as sc
def test_gh_11315(self):
    assert sc.rgamma(-35) == 0