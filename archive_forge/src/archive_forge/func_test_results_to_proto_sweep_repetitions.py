import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
def test_results_to_proto_sweep_repetitions():
    measurements = [v2.MeasureInfo('foo', [q(0, 0)], instances=1, invert_mask=[False], tags=[])]
    trial_results = [[cirq.ResultDict(params=cirq.ParamResolver({'i': 0}), records={'foo': np.array([[[0]]], dtype=bool)}), cirq.ResultDict(params=cirq.ParamResolver({'i': 1}), records={'foo': np.array([[[0]], [[1]]], dtype=bool)})]]
    with pytest.raises(ValueError, match='Different numbers of repetitions'):
        v2.results_to_proto(trial_results, measurements)