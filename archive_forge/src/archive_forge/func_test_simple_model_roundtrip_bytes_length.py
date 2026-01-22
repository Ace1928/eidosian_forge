import pytest
import srsly
from thinc.api import (
def test_simple_model_roundtrip_bytes_length():
    """Ensure that serialization of non-initialized weight matrices goes fine"""
    model1 = Maxout(5, 10, nP=2)
    model2 = Maxout(5, 10, nP=2)
    data1 = model1.to_bytes()
    model2 = model2.from_bytes(data1)
    data2 = model2.to_bytes()
    assert data1 == data2
    assert len(data1) == len(data2)