import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
def test_doi_url_not_found():
    """Should fail if the DOI is not found"""
    with pytest.raises(ValueError) as exc:
        doi_to_url(doi='NOTAREALDOI')
    assert 'Is the DOI correct?' in str(exc.value)