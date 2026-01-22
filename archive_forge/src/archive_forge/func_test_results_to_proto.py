import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
def test_results_to_proto():
    measurements = [v2.MeasureInfo('foo', [q(0, 0)], instances=1, invert_mask=[False], tags=[])]
    trial_results = [[cirq.ResultDict(params=cirq.ParamResolver({'i': 0}), records={'foo': np.array([[[0]], [[1]], [[0]], [[1]]], dtype=bool)}), cirq.ResultDict(params=cirq.ParamResolver({'i': 1}), records={'foo': np.array([[[0]], [[1]], [[1]], [[0]]], dtype=bool)})], [cirq.ResultDict(params=cirq.ParamResolver({'i': 0}), records={'foo': np.array([[[0]], [[1]], [[0]], [[1]]], dtype=bool)}), cirq.ResultDict(params=cirq.ParamResolver({'i': 1}), records={'foo': np.array([[[0]], [[1]], [[1]], [[0]]], dtype=bool)})]]
    proto = v2.results_to_proto(trial_results, measurements)
    assert isinstance(proto, v2.result_pb2.Result)
    assert len(proto.sweep_results) == 2
    deserialized = v2.results_from_proto(proto, measurements)
    assert len(deserialized) == 2
    for sweep_results, expected in zip(deserialized, trial_results):
        assert len(sweep_results) == len(expected)
        for trial_result, expected_trial_result in zip(sweep_results, expected):
            assert trial_result.params == expected_trial_result.params
            assert trial_result.repetitions == expected_trial_result.repetitions
            np.testing.assert_array_equal(trial_result.measurements['foo'], expected_trial_result.measurements['foo'])