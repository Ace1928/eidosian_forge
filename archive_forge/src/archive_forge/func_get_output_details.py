import ctypes
import enum
import os
import platform
import sys
import numpy as np
def get_output_details(self):
    """Gets model output tensor details.

    Returns:
      A list in which each item is a dictionary with details about
      an output tensor. The dictionary contains the same fields as
      described for `get_input_details()`.
    """
    return [self._get_tensor_details(i, subgraph_index=0) for i in self._interpreter.OutputIndices()]