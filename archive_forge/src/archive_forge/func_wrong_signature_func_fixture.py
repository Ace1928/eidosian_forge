import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def wrong_signature_func_fixture():
    """
    Register a scalar function with the wrong signature.
    """

    def wrong_signature():
        return pa.scalar(1, type=pa.int64())
    func_name = 'test_wrong_signature'
    in_types = {}
    out_type = pa.int64()
    doc = {'summary': 'UDF with wrong signature', 'description': ''}
    pc.register_scalar_function(wrong_signature, func_name, doc, in_types, out_type)
    return (wrong_signature, func_name)