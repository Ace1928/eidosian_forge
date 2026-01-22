from typing import Any, Dict, List, Optional, Tuple, Union
from torchgen.api.types import (
from torchgen.model import (
def isGeneratorType(typ: Type) -> bool:
    if isinstance(typ, BaseType):
        return typ.name == BaseTy.Generator
    elif isinstance(typ, OptionalType):
        return isGeneratorType(typ.elem)
    return False