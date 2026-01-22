import inspect
def check_cython(x):
    return type(x).__name__ == 'cython_function_or_method'