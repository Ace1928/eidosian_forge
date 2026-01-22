import logging; log = logging.getLogger(__name__)
import sys
from passlib.utils.decor import deprecated_method
from abc import ABCMeta, abstractmethod, abstractproperty
def recreate_with_metaclass(meta):
    """class decorator that re-creates class using metaclass"""

    def builder(cls):
        if meta is type(cls):
            return cls
        return meta(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return builder