from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import List, Optional, Union
from torchgen.model import Argument, SelfArgument, TensorOptionsArguments
def decl_registration_declarations(self) -> str:
    type_s = self.nctype.cpp_type_registration_declarations()
    mb_default = ''
    if self.default is not None:
        mb_default = f'={self.default}'
    return f'{type_s} {self.name}{mb_default}'