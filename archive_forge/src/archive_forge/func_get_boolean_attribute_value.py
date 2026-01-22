import re
import logging
import numpy as np
from .._export_onnx import MXNetGraph as mx_op
def get_boolean_attribute_value(attrs, attr_name):
    """ Helper function to convert a string version
    of Boolean attributes to integer for ONNX.
    Takes attribute dictionary and attr_name as
    parameters.
    """
    return 1 if attrs.get(attr_name, 0) in ['True', '1'] else 0