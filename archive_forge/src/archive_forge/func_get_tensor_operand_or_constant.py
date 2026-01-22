import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def get_tensor_operand_or_constant(self, jitval, dim_order=DimOrder.PRESUMED_CONTIGUOUS):
    operand_id = self.jitval_operand_map.get(jitval)
    if operand_id is None:
        _, value = self.get_constant_value(jitval, 'TensorType')
        operand_id = self.add_tensor_operand_for_weight(value, dim_order)
    return (operand_id, self.operands[operand_id])