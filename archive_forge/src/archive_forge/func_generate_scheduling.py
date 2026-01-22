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
def generate_scheduling(instruction_durations, scheduling_method, timing_constraints, inst_map, target=None):
    """Generate a post optimization scheduling :class:`~qiskit.transpiler.PassManager`

    Args:
        instruction_durations (dict): The dictionary of instruction durations
        scheduling_method (str): The scheduling method to use, can either be
            ``'asap'``/``'as_soon_as_possible'`` or
            ``'alap'``/``'as_late_as_possible'``
        timing_constraints (TimingConstraints): Hardware time alignment restrictions.
        inst_map (InstructionScheduleMap): Mapping object that maps gate to schedule.
        target (Target): The :class:`~.Target` object representing the backend

    Returns:
        PassManager: The scheduling pass manager

    Raises:
        TranspilerError: If the ``scheduling_method`` kwarg is not a valid value
    """
    scheduling = PassManager()
    if inst_map and inst_map.has_custom_gate():
        scheduling.append(PulseGates(inst_map=inst_map, target=target))
    if scheduling_method:
        scheduler = {'alap': ALAPScheduleAnalysis, 'as_late_as_possible': ALAPScheduleAnalysis, 'asap': ASAPScheduleAnalysis, 'as_soon_as_possible': ASAPScheduleAnalysis}
        scheduling.append(TimeUnitConversion(instruction_durations, target=target))
        try:
            scheduling.append(scheduler[scheduling_method](instruction_durations, target=target))
        except KeyError as ex:
            raise TranspilerError('Invalid scheduling method %s.' % scheduling_method) from ex
    elif instruction_durations:

        def _contains_delay(property_set):
            return property_set['contains_delay']
        scheduling.append(ContainsInstruction('delay'))
        scheduling.append(ConditionalController(TimeUnitConversion(instruction_durations, target=target), condition=_contains_delay))
    if timing_constraints.granularity != 1 or timing_constraints.min_length != 1 or timing_constraints.acquire_alignment != 1 or (timing_constraints.pulse_alignment != 1):

        def _require_alignment(property_set):
            return property_set['reschedule_required']
        scheduling.append(InstructionDurationCheck(acquire_alignment=timing_constraints.acquire_alignment, pulse_alignment=timing_constraints.pulse_alignment))
        scheduling.append(ConditionalController(ConstrainedReschedule(acquire_alignment=timing_constraints.acquire_alignment, pulse_alignment=timing_constraints.pulse_alignment), condition=_require_alignment))
        scheduling.append(ValidatePulseGates(granularity=timing_constraints.granularity, min_length=timing_constraints.min_length))
    if scheduling_method:
        scheduling.append(PadDelay(target=target))
    return scheduling