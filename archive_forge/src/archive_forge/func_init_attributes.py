from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab
def init_attributes(self):
    self.shape = self.get_buf_shapevars()
    self.strides = self.get_buf_stridevars()
    self.suboffsets = self.get_buf_suboffsetvars()