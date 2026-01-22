import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.path import Path
import matplotlib.patches as patches
def test_SimplePatchShadow_offset():
    pe = path_effects.SimplePatchShadow(offset=(4, 5))
    assert pe._offset == (4, 5)