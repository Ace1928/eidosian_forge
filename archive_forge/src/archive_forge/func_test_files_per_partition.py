from __future__ import annotations
from functools import partial
import pytest
from fsspec.compression import compr
from tlz import concat
from dask import compute, config
from dask.bag.text import read_text
from dask.bytes import utils
from dask.utils import filetexts
def test_files_per_partition():
    files3 = {f'{n:02}.txt': 'line from {:02}' for n in range(20)}
    with filetexts(files3):
        with config.set({'scheduler': 'single-threaded'}):
            with pytest.warns(UserWarning):
                b = read_text('*.txt', files_per_partition=10)
                l = len(b.take(100, npartitions=1))
            assert l == 10, '10 files should be grouped into one partition'
            assert b.count().compute() == 20, 'All 20 lines should be read'
            with pytest.warns(UserWarning):
                b = read_text('*.txt', files_per_partition=10, include_path=True)
                p = b.take(100, npartitions=1)
            p_paths = tuple(zip(*p))[1]
            p_unique_paths = set(p_paths)
            assert len(p_unique_paths) == 10
            b_paths = tuple(zip(*b.compute()))[1]
            b_unique_paths = set(b_paths)
            assert len(b_unique_paths) == 20