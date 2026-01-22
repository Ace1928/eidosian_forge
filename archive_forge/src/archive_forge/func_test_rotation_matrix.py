import math
from affine import Affine
def test_rotation_matrix():
    """A rotation matrix has expected elements

    | cos(a) -sin(a) |
    | sin(a)  cos(a) |

    """
    rot = Affine.rotation(90.0)
    assert round(rot.a, 15) == round(math.cos(math.pi / 2.0), 15)
    assert round(rot.b, 15) == round(-math.sin(math.pi / 2.0), 15)
    assert rot.c == 0.0
    assert round(rot.d, 15) == round(math.sin(math.pi / 2.0), 15)
    assert round(rot.e, 15) == round(math.cos(math.pi / 2.0), 15)
    assert rot.f == 0.0