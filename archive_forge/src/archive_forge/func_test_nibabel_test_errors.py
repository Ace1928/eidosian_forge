import pathlib
from unittest import mock
import pytest
import nibabel as nib
def test_nibabel_test_errors():
    with pytest.raises(NotImplementedError):
        nib.test(label='fast')
    with pytest.raises(NotImplementedError):
        nib.test(raise_warnings=[])
    with pytest.raises(NotImplementedError):
        nib.test(timer=True)
    with pytest.raises(ValueError):
        nib.test(verbose='-v')