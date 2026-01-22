from functools import wraps
from .sympify import SympifyError, sympify
def priority_decorator(func):

    @wraps(func)
    def binary_op_wrapper(self, other):
        if hasattr(other, '_op_priority'):
            if other._op_priority > self._op_priority:
                f = getattr(other, method_name, None)
                if f is not None:
                    return f(self)
        return func(self, other)
    return binary_op_wrapper