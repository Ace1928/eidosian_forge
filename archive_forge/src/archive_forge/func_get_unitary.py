import copy
import warnings
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.pulse.schedule import Schedule
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states import statevector
from qiskit.result.models import ExperimentResult
from qiskit.result import postprocess
from qiskit.result.counts import Counts
from qiskit.qobj.utils import MeasLevel
from qiskit.qobj import QobjHeader
def get_unitary(self, experiment=None, decimals=None):
    """Get the final unitary of an experiment.

        Args:
            experiment (str or QuantumCircuit or Schedule or int or None): the index of the
                experiment, as specified by ``data()``.
            decimals (int): the number of decimals in the unitary.
                If None, does not round.

        Returns:
            list[list[complex]]: list of 2^num_qubits x 2^num_qubits complex
                amplitudes.

        Raises:
            QiskitError: if there is no unitary for the experiment.
        """
    try:
        return postprocess.format_unitary(self.data(experiment)['unitary'], decimals=decimals)
    except KeyError as ex:
        raise QiskitError(f'No unitary for experiment "{repr(experiment)}"') from ex