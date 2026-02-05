import numpy as np

from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.morton import morton_encode, morton_sort


def test_morton_encode_2d_basic() -> None:
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.NONE)
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    codes = morton_encode(points, domain, bits=1)
    assert codes.tolist() == [0, 1, 2, 3]


def test_morton_sort() -> None:
    domain = Domain(mins=np.array([0.0, 0.0]), maxs=np.array([1.0, 1.0]), wrap=WrapMode.NONE)
    points = np.array(
        [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )
    _, order = morton_sort(points, domain, bits=1)
    assert order.tolist() == [1, 2, 3, 0]
