import sys
import unittest.mock as mock
import pytest
import cirq_google as cg
from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook, QCSObjectsForNotebook
@mock.patch('cirq_google.engine.qcs_notebook.get_engine')
def test_get_qcs_objects_for_notebook_mocked_engine_fails(engine_mock):
    """Tests creating an engine object which fails."""
    engine_mock.side_effect = EnvironmentError('This is a mock, not real credentials.')
    result = get_qcs_objects_for_notebook()
    _assert_correct_types(result)
    _assert_simulated_values(result)