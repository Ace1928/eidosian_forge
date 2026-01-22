import argparse
import itertools
import os
from typing import Sequence, TypeVar, Union
from libfb.py.log import set_simple_logging  # type: ignore[import]
from torchgen import gen
from torchgen.context import native_function_manager
from torchgen.model import DispatchKey, NativeFunctionsGroup, NativeFunctionsViewGroup
from torchgen.static_runtime import config, generator
def group_functions_by_op_name(grouped_native_functions: Sequence[NativeGroupT]) -> Sequence[Sequence[NativeGroupT]]:
    if not grouped_native_functions:
        return []
    groups = []

    def is_supported(g: Union[NativeFunctionsGroup, NativeFunctionsViewGroup]) -> bool:
        with native_function_manager(g):
            return generator.is_supported(g)
    eligible_ops = (g for g in grouped_native_functions if is_supported(g))
    groups = [list(group) for k, group in itertools.groupby(eligible_ops, key=config.func_name_base_str)]
    return groups