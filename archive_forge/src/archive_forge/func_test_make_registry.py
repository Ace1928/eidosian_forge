import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import pytest
from ..core import Pooch
from ..hashes import (
from .utils import check_tiny_data, mirror_directory
def test_make_registry(data_dir_mirror):
    """Check that the registry builder creates the right file names and hashes"""
    outfile = NamedTemporaryFile(delete=False)
    outfile.close()
    try:
        make_registry(data_dir_mirror, outfile.name, recursive=False)
        with open(outfile.name, encoding='utf-8') as fout:
            registry = fout.read()
        assert registry == REGISTRY
        pup = Pooch(path=data_dir_mirror, base_url='some bogus URL', registry={})
        pup.load_registry(outfile.name)
        true = str(data_dir_mirror / 'tiny-data.txt')
        fname = pup.fetch('tiny-data.txt')
        assert true == fname
        check_tiny_data(fname)
    finally:
        os.remove(outfile.name)