import codecs
import inspect
import json
import os
import re
from enum import Enum, unique
from functools import wraps
from collections.abc import Sequence
def process_file_data(cls, name, func, file_attr):
    """
    Process the parameter in the `file_data` decorator.
    """
    cls_path = os.path.abspath(inspect.getsourcefile(cls))
    data_file_path = os.path.join(os.path.dirname(cls_path), file_attr)

    def create_error_func(message):

        def func(*args):
            raise ValueError(message % file_attr)
        return func
    if not os.path.exists(data_file_path):
        test_name = mk_test_name(name, 'error')
        test_docstring = 'Error!'
        add_test(cls, test_name, test_docstring, create_error_func('%s does not exist'), None)
        return
    _is_yaml_file = data_file_path.endswith(('.yml', '.yaml'))
    if _is_yaml_file and (not _have_yaml):
        test_name = mk_test_name(name, 'error')
        test_docstring = 'Error!'
        add_test(cls, test_name, test_docstring, create_error_func('%s is a YAML file, please install PyYAML'), None)
        return
    with codecs.open(data_file_path, 'r', 'utf-8') as f:
        if _is_yaml_file:
            if hasattr(func, YAML_LOADER_ATTR):
                yaml_loader = getattr(func, YAML_LOADER_ATTR)
                data = yaml.load(f, Loader=yaml_loader)
            else:
                data = yaml.safe_load(f)
        else:
            data = json.load(f)
    _add_tests_from_data(cls, name, func, data)