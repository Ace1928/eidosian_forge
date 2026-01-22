import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def serialize_values(self):
    serialized_values = []
    serialized_value_data = []
    assert len(self.values) == len(self.value_data)
    for (op_index, source_type), data in zip(self.values, self.value_data):
        source_length = len(data)
        physical_length = (source_length - 1 | 3) + 1
        padded_data = data + b'\x00' * (physical_length - source_length)
        serialized_values.append(struct.pack('iii', op_index, source_type, source_length))
        serialized_value_data.append(padded_data)
    return (serialized_values, serialized_value_data)