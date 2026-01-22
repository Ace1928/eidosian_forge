import os
import unittest
from unittest import mock
import numpy as np
import pytest
import nibabel as nb
from nibabel.cmdline.roi import lossless_slice, main, parse_slice
from nibabel.testing import data_path
def test_lossless_slice_unknown_axes():
    img = nb.load(os.path.join(data_path, 'minc1_4d.mnc'))
    with pytest.raises(ValueError):
        lossless_slice(img, (slice(None), slice(None), slice(None)))