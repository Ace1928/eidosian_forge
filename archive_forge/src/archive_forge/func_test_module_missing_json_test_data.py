import pytest
from cirq.testing.json import spec_for
def test_module_missing_json_test_data():
    with pytest.raises(ValueError, match='json_test_data'):
        spec_for('cirq.testing.test_data.test_module_missing_json_test_data')