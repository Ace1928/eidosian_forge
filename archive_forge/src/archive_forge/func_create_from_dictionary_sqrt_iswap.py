from typing import (
import numpy as np
import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
@classmethod
def create_from_dictionary_sqrt_iswap(cls, parameters: PhasedFsimDictParameters, *, simulator: Optional[cirq.Simulator]=None, ideal_when_missing_gate: bool=False, ideal_when_missing_parameter: bool=False) -> 'PhasedFSimEngineSimulator':
    """Creates PhasedFSimEngineSimulator with fixed drifts.

        Args:
            parameters: Parameters to use for each gate. All keys must be stored in canonical order,
                when the first qubit is not greater than the second one.
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.
            ideal_when_missing_gate: When set and parameters for some gate for a given pair of
                qubits are not specified in the parameters dictionary then the
                FSimGate(theta=π/4, phi=0) gate parameters will be used. When not set and this
                situation occurs, ValueError is thrown during simulation.
            ideal_when_missing_parameter: When set and some parameter for some gate for a given pair
                of qubits is specified then the matching parameter of FSimGate(theta=π/4, phi=0)
                gate will be used. When not set and this situation occurs, ValueError is thrown
                during simulation.

        Returns:
            New PhasedFSimEngineSimulator instance.

        Raises:
            ValueError: If missing parameters for the given pair of qubits.
        """

    def sample_gate(a: cirq.Qid, b: cirq.Qid, gate: cirq.FSimGate) -> PhasedFSimCharacterization:
        _assert_inv_sqrt_iswap_like(gate)
        if (a, b) in parameters:
            pair_parameters = parameters[a, b]
            if not isinstance(pair_parameters, PhasedFSimCharacterization):
                pair_parameters = PhasedFSimCharacterization(**pair_parameters)
        elif (b, a) in parameters:
            pair_parameters = parameters[b, a]
            if not isinstance(pair_parameters, PhasedFSimCharacterization):
                pair_parameters = PhasedFSimCharacterization(**pair_parameters)
            pair_parameters = pair_parameters.parameters_for_qubits_swapped()
        elif ideal_when_missing_gate:
            pair_parameters = SQRT_ISWAP_INV_PARAMETERS
        else:
            raise ValueError(f'Missing parameters for pair {(a, b)}')
        if pair_parameters.any_none():
            if not ideal_when_missing_parameter:
                raise ValueError(f'Missing parameter value for pair {(a, b)}, parameters={pair_parameters}')
            pair_parameters = pair_parameters.merge_with(SQRT_ISWAP_INV_PARAMETERS)
        return pair_parameters
    for a, b in parameters:
        if a > b:
            raise ValueError(f'All qubit pairs must be given in canonical order where the first qubit is less than the second, got {a} > {b}')
    if simulator is None:
        simulator = cirq.Simulator()
    return cls(simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim)