import collections
from typing import Optional
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.passmanager.flow_controllers import ConditionalController
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes import Error
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import Collect1qRuns
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.transpiler.passes import HighLevelSynthesis
from qiskit.transpiler.passes import CheckMap
from qiskit.transpiler.passes import GateDirection
from qiskit.transpiler.passes import BarrierBeforeFinalMeasurements
from qiskit.transpiler.passes import CheckGateDirection
from qiskit.transpiler.passes import TimeUnitConversion
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit.transpiler.passes import ASAPScheduleAnalysis
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import RemoveResetInZeroState
from qiskit.transpiler.passes import FilterOpNodes
from qiskit.transpiler.passes import ValidatePulseGates
from qiskit.transpiler.passes import PadDelay
from qiskit.transpiler.passes import InstructionDurationCheck
from qiskit.transpiler.passes import ConstrainedReschedule
from qiskit.transpiler.passes import PulseGates
from qiskit.transpiler.passes import ContainsInstruction
from qiskit.transpiler.passes import VF2PostLayout
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayoutStopReason
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
def generate_unroll_3q(target, basis_gates=None, approximation_degree=None, unitary_synthesis_method='default', unitary_synthesis_plugin_config=None, hls_config=None):
    """Generate an unroll >3q :class:`~qiskit.transpiler.PassManager`

    Args:
        target (Target): the :class:`~.Target` object representing the backend
        basis_gates (list): A list of str gate names that represent the basis
            gates on the backend target
        approximation_degree (Optional[float]): The heuristic approximation degree to
            use. Can be between 0 and 1.
        unitary_synthesis_method (str): The unitary synthesis method to use. You can see
            a list of installed plugins with :func:`.unitary_synthesis_plugin_names`.
        unitary_synthesis_plugin_config (dict): The optional dictionary plugin
            configuration, this is plugin specific refer to the specified plugin's
            documentation for how to use.
        hls_config (HLSConfig): An optional configuration class to use for
                :class:`~qiskit.transpiler.passes.HighLevelSynthesis` pass.
                Specifies how to synthesize various high-level objects.

    Returns:
        PassManager: The unroll 3q or more pass manager
    """
    unroll_3q = PassManager()
    unroll_3q.append(UnitarySynthesis(basis_gates, approximation_degree=approximation_degree, method=unitary_synthesis_method, min_qubits=3, plugin_config=unitary_synthesis_plugin_config, target=target))
    unroll_3q.append(HighLevelSynthesis(hls_config=hls_config, coupling_map=None, target=target, use_qubit_indices=False, equivalence_library=sel, basis_gates=basis_gates, min_qubits=3))
    if basis_gates is None and target is None:
        unroll_3q.append(Unroll3qOrMore(target, basis_gates))
    else:
        unroll_3q.append(BasisTranslator(sel, basis_gates, target=target, min_qubits=3))
    return unroll_3q