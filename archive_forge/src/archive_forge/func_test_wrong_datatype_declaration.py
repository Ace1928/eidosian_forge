import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
def test_wrong_datatype_declaration():

    def identity(ctx, val):
        return val
    func_name = 'test_wrong_datatype_declaration'
    in_types = {'array': pa.int64()}
    out_type = {}
    doc = {'summary': 'test output value', 'description': 'test output'}
    with pytest.raises(TypeError, match="DataType expected, got <class 'dict'>"):
        pc.register_scalar_function(identity, func_name, doc, in_types, out_type)