import os
import unittest
from unittest import mock
import numpy as np
import pytest
import nibabel as nb
from nibabel.cmdline.roi import lossless_slice, main, parse_slice
from nibabel.testing import data_path
def test_parse_slice():
    assert parse_slice(None) == slice(None)
    assert parse_slice('1:5') == slice(1, 5)
    assert parse_slice('1:') == slice(1, None)
    assert parse_slice(':5') == slice(None, 5)
    assert parse_slice(':-1') == slice(None, -1)
    assert parse_slice('-5:-1') == slice(-5, -1)
    assert parse_slice('1:5:') == slice(1, 5, None)
    assert parse_slice('1::') == slice(1, None, None)
    assert parse_slice(':5:') == slice(None, 5, None)
    assert parse_slice(':-1:') == slice(None, -1, None)
    assert parse_slice('-5:-1:') == slice(-5, -1, None)
    assert parse_slice('1:5:1') == slice(1, 5, 1)
    assert parse_slice('1::1') == slice(1, None, 1)
    assert parse_slice(':5:1') == slice(None, 5, 1)
    assert parse_slice(':-1:1') == slice(None, -1, 1)
    assert parse_slice('-5:-1:1') == slice(-5, -1, 1)
    assert parse_slice('5:1:-1') == slice(5, 1, -1)
    assert parse_slice(':1:-1') == slice(None, 1, -1)
    assert parse_slice('5::-1') == slice(5, None, -1)
    assert parse_slice('-1::-1') == slice(-1, None, -1)
    assert parse_slice('-1:-5:-1') == slice(-1, -5, -1)
    with pytest.raises(ValueError):
        parse_slice('1:2:3:4')
    with pytest.raises(ValueError):
        parse_slice('abc:2:3')
    with pytest.raises(ValueError):
        parse_slice('1.2:2:3')
    with pytest.raises(ValueError):
        parse_slice('1:5:2')