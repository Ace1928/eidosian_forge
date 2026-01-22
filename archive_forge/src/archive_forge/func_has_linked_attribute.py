from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from onnx import load
from onnx.defs import onnx_opset_version
from onnx.external_data_helper import ExternalDataInfo, uses_external_data
from onnx.model_container import ModelContainer
from onnx.onnx_pb import (
from onnx.reference.op_run import (
from onnx.reference.ops_optimized import optimized_operators
@property
def has_linked_attribute(self):
    """Checks if the graph has a linked attribute (= an attribute whose value is defined
        by a function attribute.
        """
    return any((node.has_linked_attribute for node in self.rt_nodes_))