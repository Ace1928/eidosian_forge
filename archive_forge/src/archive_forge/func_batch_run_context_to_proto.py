from typing import cast, Dict, Iterable, List, Optional, Tuple
import sympy
import cirq
from cirq_google.api.v2 import batch_pb2
from cirq_google.api.v2 import run_context_pb2
from cirq_google.study.device_parameter import DeviceParameter
def batch_run_context_to_proto(sweepables_and_reps: Iterable[Tuple[cirq.Sweepable, int]], *, out: Optional[batch_pb2.BatchRunContext]=None) -> batch_pb2.BatchRunContext:
    """Populates a BatchRunContext protobuf message.

    Args:
        sweepables_and_reps: Iterable over tuples of (sweepable, repetitions), one
            for each run context in the batch.
        out: Optional message to be populated. If not given, a new message will
            be created.

    Returns:
        Populated BatchRunContext protobuf message.
    """
    if out is None:
        out = batch_pb2.BatchRunContext()
    for sweepable, reps in sweepables_and_reps:
        run_context_to_proto(sweepable, reps, out=out.run_contexts.add())
    return out