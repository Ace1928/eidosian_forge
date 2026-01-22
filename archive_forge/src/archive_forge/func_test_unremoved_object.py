import unittest
from unittest import mock
import pytest
from ..pkg_info import cmp_pkg_version
@mock.patch(_sched('OBJECT'), [('3.0.0', [('nibabel.nifti1', 'Nifti1Image')])])
def test_unremoved_object():
    with pytest.raises(AssertionError):
        test_object_removal()