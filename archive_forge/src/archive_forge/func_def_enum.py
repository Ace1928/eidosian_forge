import types
import weakref
import six
from apitools.base.protorpclite import util
@staticmethod
def def_enum(dct, name):
    """Define enum class from dictionary.

        Args:
          dct: Dictionary of enumerated values for type.
          name: Name of enum.
        """
    return type(name, (Enum,), dct)