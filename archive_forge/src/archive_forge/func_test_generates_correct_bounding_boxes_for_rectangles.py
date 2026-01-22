import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared._warnings import expected_warnings
from skimage.draw import random_shapes
def test_generates_correct_bounding_boxes_for_rectangles():
    image, labels = random_shapes((128, 128), max_shapes=1, shape='rectangle', rng=42)
    assert len(labels) == 1
    label, bbox = labels[0]
    assert label == 'rectangle', label
    crop = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
    assert (crop >= 0).all() and (crop < 255).all()
    image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = 255
    assert (image == 255).all()