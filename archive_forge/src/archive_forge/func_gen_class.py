import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target
def gen_class(self, f: NativeFunction, k: SchemaKind, *, class_name: str, parent_class: str, generate_super: bool) -> str:
    if k is SchemaKind.functional:
        output_type = 'Tensor'
        output_value = 'outputs_[output_idx]'
        proxy_field = ''
    elif k is SchemaKind.inplace:
        output_type = 'std::reference_wrapper<Tensor>'
        output_value = 'proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()'
        proxy_field = f'std::array<c10::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;'
    elif k is SchemaKind.out:
        output_type = 'std::reference_wrapper<Tensor>'
        output_value = 'proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()'
        proxy_field = f'std::array<c10::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;'
    if self.backend_index.dispatch_key == DispatchKey.CUDA:
        if self.rocm:
            guard_field = 'c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;'
        else:
            guard_field = 'c10::cuda::OptionalCUDAGuard guard_;'
    elif self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
        guard_field = 'c10::OptionalDeviceGuard guard_;'
    elif self.backend_index.dispatch_key == DispatchKey.MPS:
        guard_field = 'c10::OptionalDeviceGuard guard_;'
    else:
        guard_field = ''
    indent = ' ' * 4
    class_ctor_str = self.gen_class_ctor(k, class_name, len(f.func.returns))
    lines = (f'struct {class_name} final : public {parent_class} {{', f'{textwrap.indent(class_ctor_str, indent)}', f'{textwrap.indent(self.gen_class_set_output_functions(k, parent_class, generate_super), indent)}', '    const Tensor& maybe_get_output(int64_t output_idx) override {', f'      return {output_value};\n', '    }', f'    std::array<{output_type}, {len(f.func.returns)}> outputs_;', f'{textwrap.indent(proxy_field, indent)}', f'{textwrap.indent(guard_field, indent)}', '};')
    return '\n'.join((line for line in lines if line))