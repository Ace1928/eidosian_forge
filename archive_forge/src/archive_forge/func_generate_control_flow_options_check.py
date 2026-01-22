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
def generate_control_flow_options_check(layout_method=None, routing_method=None, translation_method=None, optimization_method=None, scheduling_method=None, basis_gates=(), target=None):
    """Generate a pass manager that, when run on a DAG that contains control flow, fails with an
    error message explaining the invalid options, and what could be used instead.

    Returns:
        PassManager: a pass manager that populates the ``contains_x`` properties for each of the
        control-flow operations, and raises an error if any of the given options do not support
        control flow, but a circuit with control flow is given.
    """
    bad_options = []
    message = 'Some options cannot be used with control flow.'
    for stage, given in [('layout', layout_method), ('routing', routing_method), ('translation', translation_method), ('optimization', optimization_method), ('scheduling', scheduling_method)]:
        option = stage + '_method'
        method_states = _CONTROL_FLOW_STATES[option]
        if given is not None and given in method_states.not_working:
            if method_states.working:
                message += f" Got {option}='{given}', but valid values are {list(method_states.working)}."
            else:
                message += f" Got {option}='{given}', but the entire {stage} stage is not supported."
            bad_options.append(option)
    out = PassManager()
    out.append(ContainsInstruction(CONTROL_FLOW_OP_NAMES, recurse=False))
    if bad_options:
        out.append(ConditionalController(Error(message), condition=_has_control_flow))
    backend_control = _InvalidControlFlowForBackend(basis_gates, target)
    out.append(ConditionalController(Error(backend_control.message), condition=backend_control.condition))
    return out