from __future__ import annotations
import re
import typing as T
from ..mesonlib import listify, version_compare
from ..compilers.cuda import CudaCompiler
from ..interpreter.type_checking import NoneType
from . import NewExtensionModule, ModuleInfo
from ..interpreterbase import (
@typed_pos_args('cuda.nvcc_arch_flags', (str, CudaCompiler), varargs=str)
@typed_kwargs('cuda.nvcc_arch_flags', DETECTED_KW)
def nvcc_arch_flags(self, state: 'ModuleState', args: T.Tuple[T.Union[CudaCompiler, str], T.List[str]], kwargs: ArchFlagsKwargs) -> T.List[str]:
    nvcc_arch_args = self._validate_nvcc_arch_args(args, kwargs)
    ret = self._nvcc_arch_flags(*nvcc_arch_args)[0]
    return ret