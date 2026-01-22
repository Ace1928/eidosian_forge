import importlib.util
import json
import os
import warnings
from dataclasses import dataclass, field
import torch
from ..training_args import TrainingArguments
from ..utils import cached_property, is_sagemaker_dp_enabled, logging
def is_sagemaker_model_parallel_available():
    smp_options = os.getenv('SM_HP_MP_PARAMETERS', '{}')
    try:
        smp_options = json.loads(smp_options)
        if 'partitions' not in smp_options:
            return False
    except json.JSONDecodeError:
        return False
    mpi_options = os.getenv('SM_FRAMEWORK_PARAMS', '{}')
    try:
        mpi_options = json.loads(mpi_options)
        if not mpi_options.get('sagemaker_mpi_enabled', False):
            return False
    except json.JSONDecodeError:
        return False
    return importlib.util.find_spec('smdistributed') is not None