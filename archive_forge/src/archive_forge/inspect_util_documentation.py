import inspect
Returns whether the class has a static method with the given name.

    Args:
        cls: The Python class (i.e. object of type `type`) to
            search for the method in.
        f_name: The name of the method to look up in this class
            and check whether or not it is static.
    