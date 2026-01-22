from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Union
import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig_mapping import _QCONFIG_STYLE_ORDER
from torch.ao.quantization.qconfig import QConfigAny
def set_module_name_object_type_order(self, module_name: str, object_type: Callable, index: int, qconfig_list: List[QConfigAny]) -> QConfigMultiMapping:
    """
        Set module_name QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name_object_type_order()` for more info
        """
    self._insert_qconfig_list('module_name_object_type_order_qconfigs', [module_name, object_type, index], qconfig_list)
    return self