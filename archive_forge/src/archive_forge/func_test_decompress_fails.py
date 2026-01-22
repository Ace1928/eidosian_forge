from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import pytest
from .. import Pooch
from ..processors import Unzip, Untar, Decompress
from .utils import pooch_test_url, pooch_test_registry, check_tiny_data, capture_log
@pytest.mark.network
def test_decompress_fails():
    """Should fail if method='auto' and no extension is given in the file name"""
    with TemporaryDirectory() as local_store:
        path = Path(local_store)
        pup = Pooch(path=path, base_url=BASEURL, registry=REGISTRY)
        with pytest.raises(ValueError) as exception:
            with warnings.catch_warnings():
                pup.fetch('tiny-data.txt', processor=Decompress(method='auto'))
        assert exception.value.args[0].startswith("Unrecognized file extension '.txt'")
        assert 'pooch.Unzip/Untar' not in exception.value.args[0]
        with pytest.raises(ValueError) as exception:
            with warnings.catch_warnings():
                pup.fetch('tiny-data.txt', processor=Decompress(method='bla'))
        assert exception.value.args[0].startswith("Invalid compression method 'bla'")
        assert 'pooch.Unzip/Untar' not in exception.value.args[0]
        with pytest.raises(ValueError) as exception:
            with warnings.catch_warnings():
                pup.fetch('tiny-data.txt', processor=Decompress(method='zip'))
        assert exception.value.args[0].startswith("Invalid compression method 'zip'")
        assert 'pooch.Unzip/Untar' in exception.value.args[0]
        with pytest.raises(ValueError) as exception:
            with warnings.catch_warnings():
                pup.fetch('store.zip', processor=Decompress(method='auto'))
        assert exception.value.args[0].startswith("Unrecognized file extension '.zip'")
        assert 'pooch.Unzip/Untar' in exception.value.args[0]