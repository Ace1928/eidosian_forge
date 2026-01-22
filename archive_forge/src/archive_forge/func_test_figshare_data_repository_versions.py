import os
import sys
from tempfile import TemporaryDirectory
import pytest
from .. import Pooch
from ..downloaders import (
from ..processors import Unzip
from .utils import (
@pytest.mark.network
@pytest.mark.parametrize('version, missing, present', [(1, 'LC08_L2SP_218074_20190114_20200829_02_T1-cropped.tar.gz', 'cropped-before.tar.gz'), (2, 'cropped-before.tar.gz', 'LC08_L2SP_218074_20190114_20200829_02_T1-cropped.tar.gz')])
def test_figshare_data_repository_versions(version, missing, present):
    """
    Test if setting the version in Figshare DOI works as expected
    """
    doi = f'10.6084/m9.figshare.21665630.v{version}'
    url = f'https://doi.org/{doi}/'
    figshare = FigshareRepository(doi, url)
    filenames = [item['name'] for item in figshare.api_response]
    assert present in filenames
    assert missing not in filenames