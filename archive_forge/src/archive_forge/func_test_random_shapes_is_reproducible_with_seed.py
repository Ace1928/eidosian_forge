import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage.draw import random_shapes
def test_random_shapes_is_reproducible_with_seed():
    random_seed = 42
    labels = []
    for _ in range(5):
        _, label = random_shapes((128, 128), max_shapes=5, rng=random_seed)
        labels.append(label)
    assert all((other == labels[0] for other in labels[1:]))