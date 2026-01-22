from __future__ import annotations
from types import ModuleType
from typing import Type
import numpy as np
import pandas
import pytest
import ray
import torch
from torch.utils.data import RandomSampler, Sampler, SequentialSampler
import modin.pandas as pd
from modin.experimental.torch.datasets import ModinDataLoader
@pytest.mark.parametrize('sampler_cls', [RandomSampler, SequentialSampler])
@pytest.mark.parametrize('batch_size', [16, 37])
def test_compare_dataloaders(sampler_cls: Type[Sampler], batch_size: int):
    by_modin = test_torch_dataloader(pd, sampler_cls, batch_size=batch_size)
    by_pandas = test_torch_dataloader(pandas, sampler_cls, batch_size=batch_size)
    assert len(by_modin) == len(by_pandas)
    for tensor_by_modin, tensor_by_pandas in zip(by_modin, by_pandas):
        assert np.allclose(tensor_by_modin, tensor_by_pandas), tensor_by_modin - tensor_by_pandas