import copy
import logging
import re
from typing import cast, Dict, List, Optional, Tuple
from ...config import cuda as inductor_cuda_config
from ...ir import Buffer, CUDATemplateBuffer, FixedLayout, IRNode, Layout
from ..common import IndentedBuffer
from . import cutlass_utils
from .cuda_kernel import CUDATemplateKernel
from .cuda_template import CUTLASSTemplate
from .cutlass_epilogue_gen import (
def render_evt_epilogue_declaration(self, template_output_node_name: str, evt_type_name: str, epilogue_nodes: List[IRNode]) -> str:
    """Generates the epilogue for the EVT epilogue fusion"""
    return CutlassEVTEpilogueTypeFormatter.ir_to_evt_string(template_output_node_name, evt_type_name, epilogue_nodes)