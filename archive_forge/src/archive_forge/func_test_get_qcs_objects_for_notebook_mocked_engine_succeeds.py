import sys
import unittest.mock as mock
import pytest
import cirq_google as cg
from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook, QCSObjectsForNotebook
@mock.patch('cirq_google.engine.qcs_notebook.get_engine')
def test_get_qcs_objects_for_notebook_mocked_engine_succeeds(engine_mock):
    """Uses a mocked engine call to test a 'prod' Engine."""
    fake_processor = cg.engine.SimulatedLocalProcessor(processor_id='tester', project_name='mock_project', device=cg.Sycamore)
    fake_processor2 = cg.engine.SimulatedLocalProcessor(processor_id='tester23', project_name='mock_project', device=cg.Sycamore23)
    fake_engine = cg.engine.SimulatedLocalEngine([fake_processor, fake_processor2])
    engine_mock.return_value = fake_engine
    result = get_qcs_objects_for_notebook()
    _assert_correct_types(result)
    assert result.signed_in
    assert not result.is_simulator
    assert result.project_id == 'mock_project'
    assert len(result.device.metadata.qubit_set) == 54
    result = get_qcs_objects_for_notebook(processor_id='tester')
    _assert_correct_types(result)
    assert result.signed_in
    assert not result.is_simulator
    assert result.project_id == 'mock_project'
    assert len(result.device.metadata.qubit_set) == 54
    result = get_qcs_objects_for_notebook(processor_id='tester23')
    _assert_correct_types(result)
    assert result.signed_in
    assert not result.is_simulator
    assert result.project_id == 'mock_project'
    assert len(result.device.metadata.qubit_set) == 23