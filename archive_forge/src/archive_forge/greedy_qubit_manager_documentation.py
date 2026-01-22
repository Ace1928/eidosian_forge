from typing import Iterable, List, Set, TYPE_CHECKING
from cirq.ops import named_qubit, qid_util, qubit_manager
Initializes `GreedyQubitManager`

        Args:
            prefix: The prefix to use for naming new clean ancillas allocated by the qubit manager.
                    The i'th allocated qubit is of the type `cirq.NamedQubit(f'{prefix}_{i}')`.
            size: The initial size of the pool of ancilla qubits managed by the qubit manager. The
                    qubit manager can automatically resize itself when the allocation request
                    exceeds the number of available qubits.
            maximize_reuse: Flag to control a FIFO vs LIFO strategy, defaults to False (FIFO).
        