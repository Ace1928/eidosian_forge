from typing import Iterator
import pytest
import sympy
import cirq
from cirq.study import sweeps
from cirq_google.study import DeviceParameter
from cirq_google.api import v2
@pytest.mark.parametrize('pass_out', [False, True])
def test_batch_run_context_to_proto(pass_out: bool) -> None:
    msg = v2.batch_pb2.BatchRunContext() if pass_out else None
    out = v2.batch_run_context_to_proto([], out=msg)
    if pass_out:
        assert out is msg
    assert len(out.run_contexts) == 0
    msg = v2.batch_pb2.BatchRunContext() if pass_out else None
    out = v2.batch_run_context_to_proto([(None, 10)], out=msg)
    if pass_out:
        assert out is msg
    assert len(out.run_contexts) == 1
    sweep_message = out.run_contexts[0].parameter_sweeps[0]
    assert v2.sweep_from_proto(sweep_message.sweep) == cirq.UnitSweep
    assert sweep_message.repetitions == 10
    sweep = cirq.Linspace('a', 0, 1, 21)
    msg = v2.batch_pb2.BatchRunContext() if pass_out else None
    out = v2.batch_run_context_to_proto([(None, 10), (sweep, 100)], out=msg)
    if pass_out:
        assert out is msg
    assert len(out.run_contexts) == 2
    sweep_message0 = out.run_contexts[0].parameter_sweeps[0]
    assert v2.sweep_from_proto(sweep_message0.sweep) == cirq.UnitSweep
    assert sweep_message0.repetitions == 10
    sweep_message1 = out.run_contexts[1].parameter_sweeps[0]
    assert v2.sweep_from_proto(sweep_message1.sweep) == sweep
    assert sweep_message1.repetitions == 100