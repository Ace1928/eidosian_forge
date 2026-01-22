from typing import TYPE_CHECKING
from .base import BaseOptions, BaseType
from .inputfield import InputField
from .unmountedtype import UnmountedType
from .utils import yank_fields_from_attrs
def set_input_object_type_default_value(default_value):
    """
    Change the sentinel value returned by non-specified fields in an InputObjectType
    Useful to differentiate between a field not being set and a field being set to None by using a sentinel value
    (e.g. Undefined is a good sentinel value for this purpose)

    This function should be called at the beginning of the app or in some other place where it is guaranteed to
    be called before any InputObjectType is defined.
    """
    global _INPUT_OBJECT_TYPE_DEFAULT_VALUE
    _INPUT_OBJECT_TYPE_DEFAULT_VALUE = default_value