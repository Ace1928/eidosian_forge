from minerl.herobraine.hero.spaces import Box, Dict, Discrete, MultiDiscrete, Enum
import collections
import numpy as np
def test_box_flat_map():
    b = Box(shape=[3, 2], low=-2, high=2, dtype=np.float32)
    x = b.sample()
    assert np.allclose(b.unmap(b.flat_map(x)), x)