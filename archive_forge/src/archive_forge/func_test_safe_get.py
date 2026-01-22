from io import StringIO
from os.path import join as pjoin
import numpy as np
import pytest
import nibabel as nib
from nibabel.cmdline.diff import *
from nibabel.cmdline.utils import *
from nibabel.testing import data_path
def test_safe_get():

    class TestObject:

        def __init__(self, test=None):
            self.test = test

        def get_test(self):
            return self.test
    test = TestObject()
    test.test = 2
    assert safe_get(test, 'test') == 2
    assert safe_get(test, 'failtest') == '-'