import inspect
def is_static_method(cls, f_name):
    """Returns whether the class has a static method with the given name.

    Args:
        cls: The Python class (i.e. object of type `type`) to
            search for the method in.
        f_name: The name of the method to look up in this class
            and check whether or not it is static.
    """
    for cls in inspect.getmro(cls):
        if f_name in cls.__dict__:
            return isinstance(cls.__dict__[f_name], staticmethod)
    return False