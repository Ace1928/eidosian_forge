from . import util
import numpy as np
def test_long_long_map(self):
    inp = np.ones(3)
    out = self.module.func1(inp)
    exp_out = 3
    assert out == exp_out