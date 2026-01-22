import numpy as np
import matplotlib as mpl
from matplotlib.colors import to_rgb, to_rgba
from numpy.testing import assert_array_equal
def assert_artists_equal(list1, list2):
    assert len(list1) == len(list2)
    for a1, a2 in zip(list1, list2):
        assert a1.__class__ == a2.__class__
        prop1 = a1.properties()
        prop2 = a2.properties()
        for key in USE_PROPS:
            if key not in prop1:
                continue
            v1 = prop1[key]
            v2 = prop2[key]
            if key == 'paths':
                for p1, p2 in zip(v1, v2):
                    assert_array_equal(p1.vertices, p2.vertices)
                    assert_array_equal(p1.codes, p2.codes)
            elif key == 'color':
                v1 = mpl.colors.to_rgba(v1)
                v2 = mpl.colors.to_rgba(v2)
                assert v1 == v2
            elif isinstance(v1, np.ndarray):
                assert_array_equal(v1, v2)
            else:
                assert v1 == v2