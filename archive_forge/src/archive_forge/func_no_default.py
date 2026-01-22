from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Union
from torchgen.model import Argument, SelfArgument, TensorOptionsArguments
def no_default(self) -> 'Binding':
    return Binding(name=self.name, nctype=self.nctype, default=None, argument=self.argument)