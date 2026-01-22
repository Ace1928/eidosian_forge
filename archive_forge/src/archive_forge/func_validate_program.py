from typing import Callable, Sequence, Union
from google.protobuf import any_pb2
import cirq
from cirq_google.engine.validating_sampler import VALIDATOR_TYPE
from cirq_google.serialization.serializer import Serializer
from cirq_google.api import v2
def validate_program(circuits: Sequence[cirq.AbstractCircuit], sweeps: Sequence[cirq.Sweepable], repetitions: int, serializer: Serializer, max_size: int=MAX_MESSAGE_SIZE) -> None:
    """Validate that the Program message size is below the maximum size limit.

    Args:
        circuits:  A sequence of  `cirq.Circuit` objects to validate.  For
          sweeps and runs, this will be a single circuit.  For batches,
          this will be a list of circuits.
        sweeps:  Parameters to run with each circuit.  The length of the
          sweeps sequence should be the same as the circuits argument.
        repetitions:  Number of repetitions to run with each sweep.
        serializer:  Serializer to use to serialize the circuits and sweeps.
        max_size:  proto size limit to check against.

    Raises:
        RuntimeError: if compiled proto is above the maximum size.
    """
    batch = v2.batch_pb2.BatchProgram()
    packed = any_pb2.Any()
    for circuit in circuits:
        serializer.serialize(circuit, msg=batch.programs.add())
    packed.Pack(batch)
    message_size = len(packed.SerializeToString())
    if message_size > max_size:
        raise RuntimeError('INVALID_PROGRAM: Program too long.')