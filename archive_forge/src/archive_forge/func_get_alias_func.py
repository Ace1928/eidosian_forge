import json
import warnings
from .base import string_types
def get_alias_func(base_class, nickname):
    """Get registrator function that allow aliases.

    Parameters
    ----------
    base_class : type
        base class for classes that will be reigstered
    nickname : str
        nickname of base_class for logging

    Returns
    -------
    a registrator function
    """
    register = get_register_func(base_class, nickname)

    def alias(*aliases):
        """alias registrator"""

        def reg(klass):
            """registrator function"""
            for name in aliases:
                register(klass, name)
            return klass
        return reg
    return alias