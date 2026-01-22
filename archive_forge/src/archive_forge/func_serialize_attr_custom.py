import pytest
import srsly
from thinc.api import (
@serialize_attr.register(SerializableAttr)
def serialize_attr_custom(_, value, name, model):
    return value.to_bytes()