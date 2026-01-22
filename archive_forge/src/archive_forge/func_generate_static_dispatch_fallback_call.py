import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from typing import (
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.gen_functionalization_type import (
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
from torchgen.yaml_utils import YamlDumper, YamlLoader
def generate_static_dispatch_fallback_call(sig: Union[CppSignature, DispatcherSignature], f: NativeFunction, backend_indices: List[BackendIndex]) -> str:
    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    assert cpp_sig is not None
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    ns = DEFAULT_KERNEL_NAMESPACE.replace('::native', '')
    if f.has_composite_explicit_autograd_kernel:
        return f'return {ns}::{DispatchKey.CompositeExplicitAutograd.lower()}::{name}({exprs});'
    elif f.has_composite_explicit_autograd_non_functional_kernel:
        return f'return {ns}::{DispatchKey.CompositeExplicitAutogradNonFunctional.lower()}::{name}({exprs});'
    elif f.has_composite_implicit_autograd_kernel:
        return f'return {ns}::{DispatchKey.CompositeImplicitAutograd.lower()}::{name}({exprs});'
    elif f.has_composite_implicit_autograd_nested_tensor_kernel:
        return f'return {ns}::{DispatchKey.CompositeImplicitAutogradNestedTensor.lower()}::{name}({exprs});'
    else:
        return f'TORCH_CHECK(false, "Static dispatch does not support {name} for{', '.join([str(index.dispatch_key) for index in backend_indices])} ");'