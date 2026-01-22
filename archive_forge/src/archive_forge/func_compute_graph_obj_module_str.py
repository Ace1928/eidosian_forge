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
def compute_graph_obj_module_str(data_class_str, parent_name):
    if parent_name == 'frame' and data_class_str in ['Data', 'Layout']:
        parent_parts = parent_name.split('.')
        module_str = '.'.join(['plotly.graph_objs'] + parent_parts[1:])
    elif parent_name == 'layout.template' and data_class_str == 'Layout':
        module_str = 'plotly.graph_objs'
    elif 'layout.template.data' in parent_name:
        parent_name = parent_name.replace('layout.template.data.', '')
        if parent_name:
            module_str = 'plotly.graph_objs.' + parent_name
        else:
            module_str = 'plotly.graph_objs'
    elif parent_name:
        module_str = 'plotly.graph_objs.' + parent_name
    else:
        module_str = 'plotly.graph_objs'
    return module_str