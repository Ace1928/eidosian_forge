from __future__ import annotations
import pytest
import dask
def test_make_dataset_with_processes():
    pytest.importorskip('mimesis')
    b = dask.datasets.make_people(npartitions=2)
    try:
        b.compute(scheduler='processes')
    except TypeError:
        pytest.fail('Failed to execute make_people using processes')