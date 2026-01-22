import base64
import calendar
import datetime
import sys
import six
from six.moves import urllib
from google.auth import exceptions
def copy_docstring(source_class):
    """Decorator that copies a method's docstring from another class.

    Args:
        source_class (type): The class that has the documented method.

    Returns:
        Callable: A decorator that will copy the docstring of the same
            named method in the source class to the decorated method.
    """

    def decorator(method):
        """Decorator implementation.

        Args:
            method (Callable): The method to copy the docstring to.

        Returns:
            Callable: the same method passed in with an updated docstring.

        Raises:
            google.auth.exceptions.InvalidOperation: if the method already has a docstring.
        """
        if method.__doc__:
            raise exceptions.InvalidOperation('Method already has a docstring.')
        source_method = getattr(source_class, method.__name__)
        method.__doc__ = source_method.__doc__
        return method
    return decorator