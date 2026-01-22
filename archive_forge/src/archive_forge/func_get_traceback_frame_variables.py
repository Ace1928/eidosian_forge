import functools
import inspect
import itertools
import re
import sys
import types
import warnings
from pathlib import Path
from django.conf import settings
from django.http import Http404, HttpResponse, HttpResponseNotFound
from django.template import Context, Engine, TemplateDoesNotExist
from django.template.defaultfilters import pprint
from django.urls import resolve
from django.utils import timezone
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
from django.utils.version import PY311, get_docs_version
from django.views.decorators.debug import coroutine_functions_to_sensitive_variables
def get_traceback_frame_variables(self, request, tb_frame):
    """
        Replace the values of variables marked as sensitive with
        stars (*********).
        """
    sensitive_variables = None
    if tb_frame.f_code.co_flags & inspect.CO_COROUTINE != 0 and tb_frame.f_code.co_name != 'sensitive_variables_wrapper':
        key = hash(f'{tb_frame.f_code.co_filename}:{tb_frame.f_code.co_firstlineno}')
        sensitive_variables = coroutine_functions_to_sensitive_variables.get(key, None)
    if sensitive_variables is None:
        current_frame = tb_frame
        while current_frame is not None:
            if current_frame.f_code.co_name == 'sensitive_variables_wrapper' and 'sensitive_variables_wrapper' in current_frame.f_locals:
                wrapper = current_frame.f_locals['sensitive_variables_wrapper']
                sensitive_variables = getattr(wrapper, 'sensitive_variables', None)
                break
            current_frame = current_frame.f_back
    cleansed = {}
    if self.is_active(request) and sensitive_variables:
        if sensitive_variables == '__ALL__':
            for name in tb_frame.f_locals:
                cleansed[name] = self.cleansed_substitute
        else:
            for name, value in tb_frame.f_locals.items():
                if name in sensitive_variables:
                    value = self.cleansed_substitute
                else:
                    value = self.cleanse_special_types(request, value)
                cleansed[name] = value
    else:
        for name, value in tb_frame.f_locals.items():
            cleansed[name] = self.cleanse_special_types(request, value)
    if tb_frame.f_code.co_name == 'sensitive_variables_wrapper' and 'sensitive_variables_wrapper' in tb_frame.f_locals:
        cleansed['func_args'] = self.cleansed_substitute
        cleansed['func_kwargs'] = self.cleansed_substitute
    return cleansed.items()