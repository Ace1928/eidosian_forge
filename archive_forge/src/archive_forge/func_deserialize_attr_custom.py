import pytest
import srsly
from thinc.api import (
@deserialize_attr.register(SerializableAttr)
def deserialize_attr_custom(_, value, name, model):
    return SerializableAttr().from_bytes(value)