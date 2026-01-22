from typing import Callable, Sequence, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.engine.validating_sampler import VALIDATOR_TYPE
from cirq_google.serialization.serializer import Serializer
from cirq_google.api import v2
def validate_for_engine(circuits: Sequence[cirq.AbstractCircuit], sweeps: Sequence[cirq.Sweepable], repetitions: Union[int, Sequence[int]], max_moments: int=MAX_MOMENTS, max_repetitions: int=MAX_TOTAL_REPETITIONS) -> None:
    """Validate a circuit and sweeps for sending to the Quantum Engine API.

    Args:
       circuits:  A sequence of  `cirq.Circuit` objects to validate.  For
          sweeps and runs, this will be a single circuit.  For batches,
          this will be a list of circuits.
       sweeps:  Parameters to run with each circuit.  The length of the
          sweeps sequence should be the same as the circuits argument.
       repetitions:  Number of repetitions to run with each sweep.
       max_moments: Maximum number of moments to allow.
       max_repetitions: Maximum number of parameter sweep values allowed
           when summed across all sweeps and all batches.
       max_duration_ns:  Maximum duration of the circuit, in nanoseconds.
    """
    _verify_reps(sweeps, repetitions, max_repetitions)
    _validate_depth(circuits, max_moments)
    _verify_measurements(circuits)