from typing import Any, Dict, Set, Tuple, Callable
from collections import OrderedDict
import torch
from torch.ao.quantization.fx._model_report.detector import (
from torch.ao.quantization.fx._model_report.model_report_visualizer import ModelReportVisualizer
from torch.ao.quantization.fx.graph_module import GraphModule
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.qconfig_mapping import QConfigMapping, QConfig
from torch.ao.quantization.fx._equalize import EqualizationQConfig
def generate_equalization_mapping(self) -> QConfigMapping:
    """
        Generates a QConfigMapping based on the suggestions of the
        ModelReport API for equalization. The generated mapping encompasses all the
        different types of feedback from the input-weight equalization detector.

        These configs are based on the suggestions provided by the ModelReport API
        and can only be generated once the reports have been generated.

        Returns a QConfigMapping for the equalization configuration
        """
    detector_qconfig_info_combined = self._generate_module_fqn_to_detector_info_mapping(self._update_detector_equalization_qconfig_info)
    mapping: QConfigMapping = self._generate_qconfig_mapping_helper(detector_qconfig_info_combined, self._equalization_config_generator)
    return mapping