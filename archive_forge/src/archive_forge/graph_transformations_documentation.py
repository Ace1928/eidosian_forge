import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import onnx
from onnx import ModelProto
from ..utils import logging
from .transformations_utils import (

    Convert node inputs of `Slice` nodes from int64 to int32, casting the out of range values.

    The constant node inputs are stored in `model.graph.node`, and the sole way to check which node
    they are consumed by is to iterate over nodes and check `node.input` for a match.

    Note that constant inputs to nodes as `Squeeze`, `Unsqueeze` can not be converted to int32, as the
    these operators explicitely expect int64 inputs according to ONNX specifications:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md
    