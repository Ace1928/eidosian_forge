import re
from dataclasses import dataclass
from typing import cast, Dict, List, Match, Optional, Sequence, Set, Tuple
from torchgen import local
from torchgen.api import cpp
from torchgen.api.types import BaseCType, Binding, NamedCType, tensorListT
from torchgen.model import (
from torchgen.utils import IDENT_REGEX
def replace_self_with_original_self(formula: str, postfix: str) -> str:

    def repl(m: Match[str]) -> str:
        return f'{m.group(1)}original_self{postfix}{m.group(2)}'
    return re.sub(IDENT_REGEX.format(f'self{postfix}'), repl, formula)