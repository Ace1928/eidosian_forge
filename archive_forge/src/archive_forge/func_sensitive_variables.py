import inspect
from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.http import HttpRequest
def sensitive_variables(*variables):
    """
    Indicate which variables used in the decorated function are sensitive so
    that those variables can later be treated in a special way, for example
    by hiding them when logging unhandled exceptions.

    Accept two forms:

    * with specified variable names:

        @sensitive_variables('user', 'password', 'credit_card')
        def my_function(user):
            password = user.pass_word
            credit_card = user.credit_card_number
            ...

    * without any specified variable names, in which case consider all
      variables are sensitive:

        @sensitive_variables()
        def my_function()
            ...
    """
    if len(variables) == 1 and callable(variables[0]):
        raise TypeError('sensitive_variables() must be called to use it as a decorator, e.g., use @sensitive_variables(), not @sensitive_variables.')

    def decorator(func):
        if iscoroutinefunction(func):
            sensitive_variables_wrapper = func
            wrapped_func = func
            while getattr(wrapped_func, '__wrapped__', None) is not None:
                wrapped_func = wrapped_func.__wrapped__
            try:
                file_path = inspect.getfile(wrapped_func)
            except TypeError:
                raise ValueError(f'{func.__name__} cannot safely be wrapped by @sensitive_variables, make it either non-async or defined in a Python file (not a builtin or from a native extension).')
            else:
                first_line_number = wrapped_func.__code__.co_firstlineno
                key = hash(f'{file_path}:{first_line_number}')
            if variables:
                coroutine_functions_to_sensitive_variables[key] = variables
            else:
                coroutine_functions_to_sensitive_variables[key] = '__ALL__'
        else:

            @wraps(func)
            def sensitive_variables_wrapper(*func_args, **func_kwargs):
                if variables:
                    sensitive_variables_wrapper.sensitive_variables = variables
                else:
                    sensitive_variables_wrapper.sensitive_variables = '__ALL__'
                return func(*func_args, **func_kwargs)
        return sensitive_variables_wrapper
    return decorator