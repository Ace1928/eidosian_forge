import sys
def get_named_object(module_name, member_name=None):
    """Get the Python object named by a given module and member name.

    This is usually much more convenient than dealing with ``__import__``
    directly::

        >>> doc = get_named_object('breezy.pyutils', 'get_named_object.__doc__')
        >>> doc.splitlines()[0]
        'Get the Python object named by a given module and member name.'

    :param module_name: a module name, as would be found in sys.modules if
        the module is already imported.  It may contain dots.  e.g. 'sys' or
        'os.path'.
    :param member_name: (optional) a name of an attribute in that module to
        return.  It may contain dots.  e.g. 'MyClass.some_method'.  If not
        given, the named module will be returned instead.
    :raises: ImportError or AttributeError.
    """
    if member_name:
        attr_chain = member_name.split('.')
        from_list = attr_chain[:1]
        obj = __import__(module_name, {}, {}, from_list)
        for attr in attr_chain:
            obj = getattr(obj, attr)
    else:
        __import__(module_name, globals(), locals(), [])
        obj = sys.modules[module_name]
    return obj