from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
from torchgen.api import cpp
from torchgen.api.types import Binding, CppSignature, CppSignatureGroup
from torchgen.gen import pythonify_default
from torchgen.model import (
def signature_str(self, *, skip_outputs: bool=False, symint: bool=True) -> str:
    return PythonSignature.signature_str(self, skip_outputs=skip_outputs, symint=symint) + '|deprecated'