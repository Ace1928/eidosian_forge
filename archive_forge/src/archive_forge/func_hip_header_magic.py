import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
def hip_header_magic(input_string):
    """If the file makes kernel builtin calls and does not include the cuda_runtime.h header,
    then automatically add an #include to match the "magic" includes provided by NVCC.
    TODO:
        Update logic to ignore cases where the cuda_runtime.h is included by another file.
    """
    output_string = input_string
    headers = ['hip/hip_runtime.h', 'hip/hip_runtime_api.h']
    if any((re.search(f'#include ("{ext}"|<{ext}>)', output_string) for ext in headers)):
        return output_string
    hasDeviceLogic: int
    hasDeviceLogic = 'hipLaunchKernelGGL' in output_string
    hasDeviceLogic += '__global__' in output_string
    hasDeviceLogic += '__shared__' in output_string
    hasDeviceLogic += RE_SYNCTHREADS.search(output_string) is not None
    if hasDeviceLogic:
        output_string = '#include "hip/hip_runtime.h"\n' + input_string
    return output_string