import types
import weakref
import six
from apitools.base.protorpclite import util
@classmethod
def lookup_field_type_by_variant(cls, variant):
    return cls.__variant_to_type[variant]