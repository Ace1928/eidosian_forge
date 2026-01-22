from typing import Any, Dict, Set, Tuple, Callable, List
import torch
import torch.nn as nn
import torch.ao.nn.qat as nnqat
from abc import ABC, abstractmethod
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.fx._model_report.model_report_observer import ModelReportObserver
from torch.ao.quantization.qconfig import (
from torch.ao.quantization.observer import (
from torch.ao.quantization.fx._equalize import (
from torch.ao.quantization.observer import _is_activation_post_process
def generate_quantization_qconfig(self, module: torch.nn.Module) -> QConfig:
    """
        Args:
            module (torch.nn.Module) The module we are generating
            the qconfig for

        Returns the generated quantization QConfig according to what a valid configuration is
        """
    module_qconfig = default_qconfig
    recommendations_list = []
    recommendations_list.append((self.is_activation_dynamic, self.is_weight_per_channel))
    recommendations_list.append((self.is_activation_dynamic, False))
    recommendations_list.append((False, self.is_weight_per_channel))
    for rec in recommendations_list:
        activation = default_dynamic_quant_observer if rec[0] else default_observer
        weight = default_per_channel_weight_observer if rec[1] else default_weight_observer
        test_config = QConfig(activation, weight)
        try:
            _assert_valid_qconfig(test_config, module)
            module_qconfig = test_config
            break
        except AssertionError:
            continue
    return module_qconfig