import os
from nibabel.optpkg import optional_package
import pytest
from nipype.utils.provenance import ProvStore, safe_encode
def test_safe_encode():
    a = 'Ã©lg'
    out = safe_encode(a)
    assert out.value == a