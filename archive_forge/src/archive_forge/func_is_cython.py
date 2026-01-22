import inspect
def is_cython(obj):
    """Check if an object is a Cython function or method"""

    def check_cython(x):
        return type(x).__name__ == 'cython_function_or_method'
    return check_cython(obj) or (hasattr(obj, '__func__') and check_cython(obj.__func__))