import collections
import contextlib
import dataclasses
import functools
import inspect
import os
import re
from itertools import chain, count
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import sympy
from sympy import Expr
import torch
from torch._dynamo.utils import counters, dynamo_timed
from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.fx.node import _get_qualified_name
from torch.utils._sympy.singleton_int import SingletonInt
from .. import codecache, config, ir
from ..codecache import CudaKernelParamCache
from ..ir import ComputedBuffer, InputBuffer, ReinterpretView
from ..triton_heuristics import grid as default_grid
from ..utils import (
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, PythonPrinter
from .triton_utils import config_of, signature_to_meta
def write_wrapper_decl(self):
    inputs_len = len(V.graph.graph_inputs.keys())
    if V.graph.aot_mode:
        self.prefix.splice('\n                void AOTInductorModel::run_impl(\n                    AtenTensorHandle*\n                        input_handles, // array of input AtenTensorHandle; handles\n                                        // are stolen; the array itself is borrowed\n                    AtenTensorHandle*\n                        output_handles, // array for writing output AtenTensorHandle; handles\n                                        // will be stolen by the caller; the array itself is\n                                        // borrowed\n                    DeviceStreamType stream,\n                    AOTIProxyExecutorHandle proxy_executor\n                ) {\n                ')
    else:
        self.prefix.splice(f'std::vector<at::Tensor> {self.call_func_name}(const std::vector<at::Tensor>& inputs) {{')
    with self.prefix.indent():
        if V.graph.aot_mode:
            if config.aot_inductor.abi_compatible:
                self.prefix.splice('\n                            auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, num_inputs());\n                        ')
            else:
                self.prefix.splice('\n                            auto inputs = alloc_tensors_by_stealing_from_handles(input_handles, num_inputs());\n                        ')
        else:
            self.prefix.splice('\n                        py::gil_scoped_release release;\n                    ')
        if inputs_len != 0:
            for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
                if isinstance(V.graph.graph_inputs[input_key], sympy.Expr):
                    from ..graph import may_get_constant_buffer_dtype
                    from .cpp import DTYPE_TO_CPP
                    dtype = may_get_constant_buffer_dtype(V.graph.graph_inputs[input_key])
                    assert dtype is not None, 'Fails to get the dtype of the sympy.Expr'
                    cpp_dtype = DTYPE_TO_CPP[dtype]
                    assert not config.aot_inductor.abi_compatible, 'Need to add .item support for abi_compatible AOTInductor codegen'
                    self.prefix.writeline(f'{cpp_dtype} {input_key} = inputs[{idx}].item<{cpp_dtype}>();')
                else:
                    self.prefix.writeline(f'auto {input_key} = std::move(inputs[{idx}]);')
        assert all((isinstance(v, torch.Tensor) for v in list(V.graph.constants.values()))), 'Expect all constants to be Tensor'
        for idx, constants_key in enumerate(V.graph.constants.keys()):
            if V.graph.aot_mode:
                if config.aot_inductor.abi_compatible:
                    self.prefix.writeline(f'auto {constants_key} = constants_.at({idx});')
                else:
                    self.prefix.writeline(f'auto {constants_key} = *tensor_handle_to_tensor_pointer(' + f'constants_.at({idx}));')
            else:
                constants_idx = inputs_len + idx
                self.prefix.writeline(f'auto {constants_key} = inputs[{constants_idx}];')
        self.codegen_inputs(self.prefix, V.graph.graph_inputs)
        if V.graph.aot_mode:
            self.prefix.writeline('inputs.clear();')
            self.prefix.writeline('auto& kernels = *dynamic_cast<AOTInductorModelKernels*>(this->kernels_.get());')