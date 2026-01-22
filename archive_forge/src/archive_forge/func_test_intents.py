import os
import struct
import unittest
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from nibabel import nifti1 as nifti1
from nibabel.affines import from_matvec
from nibabel.casting import have_binary128, type_info
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import (
from nibabel.optpkg import optional_package
from nibabel.pkg_info import cmp_pkg_version
from nibabel.spatialimages import HeaderDataError
from nibabel.tmpdirs import InTemporaryDirectory
from ..freesurfer import load as mghload
from ..orientations import aff2axcodes
from ..testing import (
from . import test_analyze as tana
from . import test_spm99analyze as tspm
from .nibabel_data import get_nibabel_data, needs_nibabel_data
from .test_arraywriters import IUINT_TYPES, rt_err_estimate
from .test_orientations import ALL_ORNTS
def test_intents(self):
    ehdr = self.header_class()
    ehdr.set_intent('t test', (10,), name='some score')
    assert ehdr.get_intent() == ('t test', (10.0,), 'some score')
    with pytest.raises(KeyError):
        ehdr.set_intent('no intention')
    with pytest.raises(KeyError):
        ehdr.set_intent('no intention', allow_unknown=True)
    with pytest.raises(KeyError):
        ehdr.set_intent(32767)
    with pytest.raises(HeaderDataError):
        ehdr.set_intent('t test', (10, 10))
    with pytest.raises(HeaderDataError):
        ehdr.set_intent('f test', (10,))
    ehdr.set_intent('t test')
    assert (ehdr['intent_p1'], ehdr['intent_p2'], ehdr['intent_p3']) == (0, 0, 0)
    assert ehdr['intent_name'] == b''
    ehdr.set_intent('t test', (10,))
    assert (ehdr['intent_p2'], ehdr['intent_p3']) == (0, 0)
    ehdr.set_intent(9999, allow_unknown=True)
    assert ehdr.get_intent() == ('unknown code 9999', (), '')
    assert ehdr.get_intent('code') == (9999, (), '')
    ehdr.set_intent(9999, name='custom intent', allow_unknown=True)
    assert ehdr.get_intent() == ('unknown code 9999', (), 'custom intent')
    assert ehdr.get_intent('code') == (9999, (), 'custom intent')
    ehdr.set_intent(code=9999, params=(1, 2, 3), allow_unknown=True)
    assert ehdr.get_intent() == ('unknown code 9999', (), '')
    assert ehdr.get_intent('code') == (9999, (), '')
    with pytest.raises(HeaderDataError):
        ehdr.set_intent(999, (1,), allow_unknown=True)
    with pytest.raises(HeaderDataError):
        ehdr.set_intent(999, (1, 2), allow_unknown=True)