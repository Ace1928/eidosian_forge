from __future__ import annotations
import itertools
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build
from .. import mesonlib
from ..interpreter.type_checking import CT_INPUT_KW
from ..interpreterbase.decorators import KwargInfo, typed_kwargs, typed_pos_args
def detect_tools(self, state: ModuleState) -> None:
    self.tools['yosys'] = state.find_program('yosys')
    self.tools['arachne'] = state.find_program('arachne-pnr')
    self.tools['icepack'] = state.find_program('icepack')
    self.tools['iceprog'] = state.find_program('iceprog')
    self.tools['icetime'] = state.find_program('icetime')