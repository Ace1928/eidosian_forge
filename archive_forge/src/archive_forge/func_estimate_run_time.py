from typing import List, Optional, Sequence
import cirq
def estimate_run_time(program: cirq.AbstractCircuit, repetitions: int, latency: Optional[float]=_BASE_LATENCY) -> float:
    """Compute the estimated time for running a single circuit.

    This should approximate, in seconds, the time for the execution of a batch of circuits
    using Engine.run() on QCS at a time where there is no queue (such as a reserved slot).
    This estimation should be considered a rough approximation.  Many factors can contribute to
    the execution time of a circuit, and the run time can also vary as the service's code changes
    frequently.

    Args:
        program: circuit to be executed
        repetitions: number of repetitions to execute
        latency: Optional latency to add (defaults to 1.5 seconds)
    """
    width = len(program.all_qubits())
    depth = len(program)
    return _estimate_run_time_seconds(width, depth, 1, repetitions, latency)