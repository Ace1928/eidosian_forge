import argparse
import os
import pathlib
import re
from collections import Counter, namedtuple
from typing import (
import yaml
import torchgen.dest as dest
from torchgen.api.lazy import setValueT
from torchgen.api.types import BaseCppType
from torchgen.dest.lazy_ir import GenLazyIR, GenLazyNativeFuncDefinition, GenTSLazyIR
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import NativeFunction, NativeFunctionsGroup, OperatorName
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, FileManager, NamespaceHelper
from torchgen.yaml_utils import YamlLoader
from .gen_backend_stubs import (
def concat_map_codegen(func: Callable[[NativeFunction], Sequence[str]], xs: Iterable[Union[NativeFunctionsGroup, NativeFunction]], ops_list: List[OperatorName]=full_codegen) -> Iterator[str]:
    """
        We code-gen for the functional variant, which is all we need for IR classes/lowerings/shape inferences, but we
        only code-gen additional entries for the inplace variant for the native functions.
        """
    for x in xs:
        fs = list(x.functions()) if isinstance(x, NativeFunctionsGroup) else [x]
        for f in fs:
            if f.func.name in ops_list:
                yield from func(f)