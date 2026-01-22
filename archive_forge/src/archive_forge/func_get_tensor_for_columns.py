import os
import warnings
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
import torch
import ray
from ray.air.util.data_batch_conversion import _unwrap_ndarray_object_type_if_needed
def get_tensor_for_columns(columns, dtype):
    feature_tensors = []
    if columns:
        batch = data_batch[columns]
    else:
        batch = data_batch
    for col in batch.columns:
        col_vals = batch[col].values
        try:
            t = tensorize(col_vals, dtype=dtype)
        except Exception:
            raise ValueError(f'Failed to convert column {col} to a Torch Tensor of dtype {dtype}. See above exception chain for the exact failure.')
        if unsqueeze:
            t = t.unsqueeze(1)
        feature_tensors.append(t)
    if len(feature_tensors) > 1:
        feature_tensor = torch.cat(feature_tensors, dim=1)
    else:
        feature_tensor = feature_tensors[0]
    return feature_tensor