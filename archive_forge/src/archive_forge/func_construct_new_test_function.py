import functools
import types
def construct_new_test_function(original_func, name, build_params):
    """Builds a new test function based on parameterized data.

    :param original_func: The original test function that is used as a template
    :param name: The fullname of the new test function
    :param build_params: A dictionary or list containing args or kwargs
        for the new test
    :return: A new function object
    """
    new_func = types.FunctionType(original_func.__code__, original_func.__globals__, name=name, argdefs=original_func.__defaults__)
    for key, val in original_func.__dict__.items():
        if key != 'build_data':
            new_func.__dict__[key] = val
    build_args = build_params if isinstance(build_params, list) else []
    build_kwargs = build_params if isinstance(build_params, dict) else {}

    def test_wrapper(func, test_args, test_kwargs):

        @functools.wraps(func)
        def wrapper(self):
            return func(self, *test_args, **test_kwargs)
        return wrapper
    return test_wrapper(new_func, build_args, build_kwargs)