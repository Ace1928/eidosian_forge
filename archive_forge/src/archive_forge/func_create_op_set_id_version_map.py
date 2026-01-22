import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def create_op_set_id_version_map(table: VersionTableType) -> VersionMapType:
    """Create a map from (opset-domain, opset-version) to ir-version from above table."""
    result: VersionMapType = {}

    def process(release_version: str, ir_version: int, *args: Any) -> None:
        del release_version
        for pair in zip(['ai.onnx', 'ai.onnx.ml', 'ai.onnx.training'], args):
            if pair not in result:
                result[pair] = ir_version
                if pair[0] == 'ai.onnx.training':
                    result['ai.onnx.preview.training', pair[1]] = ir_version
    for row in table:
        process(*row)
    return result