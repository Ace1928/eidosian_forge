from typing import List, Optional, Set, Tuple, Union
import dcgm_fields
import torch
from dcgm_fields import DcgmFieldGetById
from dcgm_structs import DCGM_GROUP_EMPTY, DCGM_OPERATION_MODE_AUTO
from pydcgm import DcgmFieldGroup, DcgmGroup, DcgmHandle
from .profiler import _Profiler, logger

        Args:
            main_profiler: The main profiler object.
            gpus_to_profile: A tuple of integers representing the GPUs to profile. If `None`,
                then the "default" GPU is used.
            field_ids_to_profile:
                See https://github.com/NVIDIA/DCGM/blob/master/testing/python3/dcgm_fields.py#L436
                for a full list of available fields. Note that not all fields are profilable.
            updateFreq: The interval of two consecutive updates of each field. Defaults to 5000 microseconds.
                This is a good tradeoff between performance and accuracy.
                An even smaller updateFreq is not supported well by A100.
                If the step to profile takes more than 5000 microseconds, then a larger updateFreq could also be used.
        