import os
import unittest
from unittest import mock
import numpy as np
import pytest
import nibabel as nb
from nibabel.cmdline.roi import lossless_slice, main, parse_slice
from nibabel.testing import data_path
def test_parse_slice_disallow_step():
    assert parse_slice('1:5', False) == slice(1, 5)
    assert parse_slice('1:5:', False) == slice(1, 5)
    assert parse_slice('1:5:1', False) == slice(1, 5, 1)
    with pytest.raises(ValueError):
        parse_slice('1:5:-1', False)
    with pytest.raises(ValueError):
        parse_slice('1:5:-2', False)