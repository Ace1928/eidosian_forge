def get_mod_func(callback):
    """
    splits the string path to a class into a string path to the module
    and the name of the class.

    Examples
    ========

    >>> from sympy.utilities.source import get_mod_func
    >>> get_mod_func('sympy.core.basic.Basic')
    ('sympy.core.basic', 'Basic')

    """
    dot = callback.rfind('.')
    if dot == -1:
        return (callback, '')
    return (callback[:dot], callback[dot + 1:])