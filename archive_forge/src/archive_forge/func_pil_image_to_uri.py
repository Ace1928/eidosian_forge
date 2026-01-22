import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
@staticmethod
def pil_image_to_uri(v):
    in_mem_file = io.BytesIO()
    v.save(in_mem_file, format='PNG')
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    base64_encoded_result_bytes = base64.b64encode(img_bytes)
    base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')
    v = 'data:image/png;base64,{base64_encoded_result_str}'.format(base64_encoded_result_str=base64_encoded_result_str)
    return v