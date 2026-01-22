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
@pytest.mark.parametrize('lib', [pandas, pd])
@pytest.mark.parametrize('sampler_cls', [RandomSampler, SequentialSampler])
@pytest.mark.parametrize('batch_size', [16, 37])
def test_torch_dataloader(lib: ModuleType, sampler_cls: Type[Sampler], batch_size: int):
    df = _load_test_dataframe(lib)
    np.random.seed(42)
    torch.manual_seed(42)
    loader = ModinDataLoader(df, batch_size=batch_size, features=['AVG_AREA_INCOME', 'AVG_AREA_HOUSE_AGE', 'AVG_AREA_NUM_ROOMS', 'AVG_AREA_NUM_BEDROOMS', 'POPULATION', 'PRICE'], sampler=sampler_cls)
    outputs = []
    for batch in loader:
        assert batch.shape[0] <= batch_size, batch.shape
        assert batch.shape[1] == 6, batch.shape
        outputs.append(batch)
    return outputs