import datetime
import pytest
import pyarrow as pa
@pytest.mark.gandiva
def test_get_registered_function_signatures():
    import pyarrow.gandiva as gandiva
    signatures = gandiva.get_registered_function_signatures()
    assert type(signatures[0].return_type()) is pa.DataType
    assert type(signatures[0].param_types()) is list
    assert hasattr(signatures[0], 'name')