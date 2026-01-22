import sys
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Optional, Sequence, Type, TypeVar, Union
@abstractmethod
def on_state(self, inputs: T_FsmInputs) -> None:
    ...