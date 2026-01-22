import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.parametrize('repository,doi', [(FigshareRepository, '10.6084/m9.figshare.14763051.v1'), (ZenodoRepository, '10.5281/zenodo.4924875'), (DataverseRepository, '10.11588/data/TKCFEF')], ids=['figshare', 'zenodo', 'dataverse'])
def test_figshare_url_file_not_found(repository, doi):
    """Should fail if the file is not found in the archive"""
    with pytest.raises(ValueError) as exc:
        url = doi_to_url(doi)
        repo = repository.initialize(doi, url)
        repo.download_url(file_name='bla.txt')
    assert "File 'bla.txt' not found" in str(exc.value)