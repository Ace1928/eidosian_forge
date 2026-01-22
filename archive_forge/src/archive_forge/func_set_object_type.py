from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Union
import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig_mapping import _QCONFIG_STYLE_ORDER
from torch.ao.quantization.qconfig import QConfigAny
def set_object_type(self, object_type: Union[Callable, str], qconfig_list: List[QConfigAny]) -> QConfigMultiMapping:
    """
        Set object type QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_object_type()` for more info
        """
    self._insert_qconfig_list('object_type_qconfigs', [object_type], qconfig_list)
    return self