import sys
import unittest.mock as mock
import pytest
import cirq_google as cg
from cirq_google.engine.qcs_notebook import get_qcs_objects_for_notebook, QCSObjectsForNotebook
@mock.patch('cirq_google.engine.qcs_notebook.get_engine')
def test_get_qcs_objects_for_notebook_no_processors(engine_mock):
    fake_engine = cg.engine.SimulatedLocalEngine([])
    engine_mock.return_value = fake_engine
    with pytest.raises(ValueError, match='processors'):
        _ = get_qcs_objects_for_notebook()