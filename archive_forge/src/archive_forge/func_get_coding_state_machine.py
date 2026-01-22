import logging
from .codingstatemachinedict import CodingStateMachineDict
from .enums import MachineState
def get_coding_state_machine(self) -> str:
    return self._model['name']