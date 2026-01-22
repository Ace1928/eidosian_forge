from typing import Any, Dict, List
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap, PulseQobjDef
from qiskit.qobj import PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.converters import QobjToInstructionConverter
Create a new PulseDefaults object from a dictionary.

        Args:
            data (dict): A dictionary representing the PulseDefaults
                         to create. It will be in the same format as output by
                         :meth:`to_dict`.
        Returns:
            PulseDefaults: The PulseDefaults from the input dictionary.
        