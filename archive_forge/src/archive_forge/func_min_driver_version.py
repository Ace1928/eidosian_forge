from __future__ import annotations
import re
import typing as T
from ..mesonlib import listify, version_compare
from ..compilers.cuda import CudaCompiler
from ..interpreter.type_checking import NoneType
from . import NewExtensionModule, ModuleInfo
from ..interpreterbase import (
@noKwargs
def min_driver_version(self, state: 'ModuleState', args: T.List[TYPE_var], kwargs: T.Dict[str, T.Any]) -> str:
    argerror = InvalidArguments('min_driver_version must have exactly one positional argument: ' + 'a CUDA Toolkit version string. Beware that, since CUDA 11.0, ' + "the CUDA Toolkit's components (including NVCC) are versioned " + 'independently from each other (and the CUDA Toolkit as a whole).')
    if len(args) != 1 or not isinstance(args[0], str):
        raise argerror
    cuda_version = args[0]
    driver_version_table = [{'cuda_version': '>=12.0.0', 'windows': '527.41', 'linux': '525.60.13'}, {'cuda_version': '>=11.8.0', 'windows': '522.06', 'linux': '520.61.05'}, {'cuda_version': '>=11.7.1', 'windows': '516.31', 'linux': '515.48.07'}, {'cuda_version': '>=11.7.0', 'windows': '516.01', 'linux': '515.43.04'}, {'cuda_version': '>=11.6.1', 'windows': '511.65', 'linux': '510.47.03'}, {'cuda_version': '>=11.6.0', 'windows': '511.23', 'linux': '510.39.01'}, {'cuda_version': '>=11.5.1', 'windows': '496.13', 'linux': '495.29.05'}, {'cuda_version': '>=11.5.0', 'windows': '496.04', 'linux': '495.29.05'}, {'cuda_version': '>=11.4.3', 'windows': '472.50', 'linux': '470.82.01'}, {'cuda_version': '>=11.4.1', 'windows': '471.41', 'linux': '470.57.02'}, {'cuda_version': '>=11.4.0', 'windows': '471.11', 'linux': '470.42.01'}, {'cuda_version': '>=11.3.0', 'windows': '465.89', 'linux': '465.19.01'}, {'cuda_version': '>=11.2.2', 'windows': '461.33', 'linux': '460.32.03'}, {'cuda_version': '>=11.2.1', 'windows': '461.09', 'linux': '460.32.03'}, {'cuda_version': '>=11.2.0', 'windows': '460.82', 'linux': '460.27.03'}, {'cuda_version': '>=11.1.1', 'windows': '456.81', 'linux': '455.32'}, {'cuda_version': '>=11.1.0', 'windows': '456.38', 'linux': '455.23'}, {'cuda_version': '>=11.0.3', 'windows': '451.82', 'linux': '450.51.06'}, {'cuda_version': '>=11.0.2', 'windows': '451.48', 'linux': '450.51.05'}, {'cuda_version': '>=11.0.1', 'windows': '451.22', 'linux': '450.36.06'}, {'cuda_version': '>=10.2.89', 'windows': '441.22', 'linux': '440.33'}, {'cuda_version': '>=10.1.105', 'windows': '418.96', 'linux': '418.39'}, {'cuda_version': '>=10.0.130', 'windows': '411.31', 'linux': '410.48'}, {'cuda_version': '>=9.2.148', 'windows': '398.26', 'linux': '396.37'}, {'cuda_version': '>=9.2.88', 'windows': '397.44', 'linux': '396.26'}, {'cuda_version': '>=9.1.85', 'windows': '391.29', 'linux': '390.46'}, {'cuda_version': '>=9.0.76', 'windows': '385.54', 'linux': '384.81'}, {'cuda_version': '>=8.0.61', 'windows': '376.51', 'linux': '375.26'}, {'cuda_version': '>=8.0.44', 'windows': '369.30', 'linux': '367.48'}, {'cuda_version': '>=7.5.16', 'windows': '353.66', 'linux': '352.31'}, {'cuda_version': '>=7.0.28', 'windows': '347.62', 'linux': '346.46'}]
    driver_version = 'unknown'
    for d in driver_version_table:
        if version_compare(cuda_version, d['cuda_version']):
            driver_version = d.get(state.environment.machines.host.system, d['linux'])
            break
    return driver_version