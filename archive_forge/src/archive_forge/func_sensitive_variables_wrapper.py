import inspect
from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.http import HttpRequest
@wraps(func)
def sensitive_variables_wrapper(*func_args, **func_kwargs):
    if variables:
        sensitive_variables_wrapper.sensitive_variables = variables
    else:
        sensitive_variables_wrapper.sensitive_variables = '__ALL__'
    return func(*func_args, **func_kwargs)