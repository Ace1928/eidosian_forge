from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type, Callable
from torch.ao.quantization.quant_type import QuantType
import torch
import copy
import warnings
from torch.fx import (
from torch.fx.graph import (
from ..utils import (
from ..qconfig import (
from ..qconfig_mapping import QConfigMapping
from .qconfig_mapping_utils import (
from torch.ao.quantization.backend_config.utils import (
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.observer import _is_activation_post_process
from .graph_module import (
from ._equalize import update_obs_for_equalization, convert_eq_obs
from torch.nn.utils.parametrize import type_before_parametrizations
from .utils import (
from torch.ao.quantization.utils import (
from torch.ao.quantization.quantize import (
from torch.ao.quantization.stubs import DeQuantStub
from .custom_config import (
from .lower_to_fbgemm import lower_to_fbgemm
from ._decomposed import quantized_decomposed_lib  # noqa: F401
import operator
def convert_weighted_module(node: Node, modules: Dict[str, torch.nn.Module], observed_node_names: Set[str], node_name_to_qconfig: Dict[str, QConfigAny], backend_config: BackendConfig, is_decomposed: bool=False, is_reference: bool=False) -> None:
    """ Convert a weighted module to reference quantized module in the model
    If the QConfig of a QAT module is not set, the module will still be converted to
    a float module.

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - observed_node_names: names for the set of observed fx node, we can skip
        this conversion if the node is not observed
    """
    original_module = modules[str(node.target)]
    qconfig: QConfigAny = original_module.qconfig
    weight_post_process = None
    qat_module_classes = get_qat_module_classes(backend_config)
    if isinstance(original_module, qat_module_classes):
        weight_post_process = original_module.weight_fake_quant
        original_module = original_module.to_float()
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, original_module)
    is_observed = node.name in observed_node_names
    if qconfig is None or _has_none_qconfig(node, node_name_to_qconfig) or (not is_observed):
        return
    pattern_to_dtype_configs = get_pattern_to_dtype_configs(backend_config)
    dtype_configs = pattern_to_dtype_configs.get(type(original_module), [])
    if not _is_qconfig_supported_by_dtype_configs(qconfig, dtype_configs):
        return
    is_weight_quantized = weight_is_quantized(qconfig)
    if not is_weight_quantized:
        return
    fused_module = None
    float_module = original_module
    if isinstance(original_module, torch.ao.nn.intrinsic._FusedModule):
        fused_module = float_module
        float_module = fused_module[0]
    wq_or_wq_dict = {'is_decomposed': is_decomposed}
    if isinstance(float_module, torch.nn.RNNCellBase):
        weight_post_process_ih = qconfig.weight()
        weight_post_process_hh = qconfig.weight()
        weight_post_process_ih(float_module.weight_ih)
        weight_post_process_hh(float_module.weight_hh)
        weight_qparams_ih = get_qparam_dict(weight_post_process_ih)
        weight_qparams_hh = get_qparam_dict(weight_post_process_hh)
        wq_or_wq_dict.update({'weight_ih': weight_qparams_ih, 'weight_hh': weight_qparams_hh})
    elif isinstance(float_module, (torch.nn.LSTM, torch.nn.GRU)):
        for wn in float_module._flat_weights_names:
            if hasattr(float_module, wn) and wn.startswith('weight'):
                weight = getattr(float_module, wn)
                weight_post_process = qconfig.weight()
                if weight_post_process.dtype == torch.qint8:
                    weight_post_process(weight)
                wq_or_wq_dict[wn] = get_qparam_dict(weight_post_process)
    else:
        is_ptq = weight_post_process is None
        if is_ptq:
            weight_post_process = qconfig.weight()
            device = assert_and_get_unique_device(float_module)
            if device:
                weight_post_process.to(device)
        is_qat = not is_ptq
        if not (is_decomposed and is_reference and is_qat):
            weight_post_process(float_module.weight)
        wq_or_wq_dict.update(get_qparam_dict(weight_post_process))
    root_module_to_quantized_reference_module = get_root_module_to_quantized_reference_module(backend_config)
    ref_qmodule_cls = root_module_to_quantized_reference_module.get(type_before_parametrizations(float_module), None)
    assert ref_qmodule_cls is not None, f'No reference quantized module class configured for {type_before_parametrizations(float_module)}'
    ref_qmodule = ref_qmodule_cls.from_float(float_module, wq_or_wq_dict)
    if fused_module is not None:
        fused_module[0] = ref_qmodule
    else:
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, ref_qmodule)