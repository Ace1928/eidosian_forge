import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
@staticmethod
def get_cached_instance_for_options(options: Optional[Union[OrtBackendOptions, Mapping[str, Any]]]=None) -> 'OrtBackend':
    """Returns a possibly cached instance of an ``OrtBackend``. If an existing
        backend was created previously through this function with the same options,
        it will be returned. Otherwise a new backend will be created, cached, and
        returned.

        Note: if ``options`` sets ``ort_session_options``, a new ``OrtBackend``
        will always be returned, since ``onnxruntime.SessionOptions`` cannot
        participate in caching."""

    def reusable(a: OrtBackendOptions, b: OrtBackendOptions):
        if a.preferred_execution_providers != b.preferred_execution_providers or a.infer_execution_providers != b.infer_execution_providers or a.default_execution_providers != b.default_execution_providers or (a.preallocate_output != b.preallocate_output) or (a.use_aot_autograd != b.use_aot_autograd):
            return False
        if a.ort_session_options is not None or b.ort_session_options is not None:
            return False
        if a.export_options is b.export_options:
            return True
        if a.export_options is not None and b.export_options is not None:
            return a.export_options.dynamic_shapes == b.export_options.dynamic_shapes and a.export_options.op_level_debug == b.export_options.op_level_debug and (a.export_options.diagnostic_options == b.export_options.diagnostic_options) and (a.export_options.onnx_registry is b.export_options.onnx_registry) and (a.export_options.fake_context is b.export_options.fake_context)
        return False
    if not isinstance(options, OrtBackendOptions):
        options = OrtBackendOptions(**options or {})
    backend = next((b for b in OrtBackend.__instance_cache if reusable(b._options, options)), None)
    if backend is None:
        assert len(OrtBackend.__instance_cache) < OrtBackend.__instance_cache_max_count, f'No more than {OrtBackend.__instance_cache_max_count} instances of {OrtBackend} allowed. Please instantiate `{OrtBackend}` explicitly to pass to `torch.compile`. See https://github.com/pytorch/pytorch/pull/107973#discussion_r1306144795 for discussion.'
        OrtBackend.__instance_cache.append((backend := OrtBackend(options)))
    return backend