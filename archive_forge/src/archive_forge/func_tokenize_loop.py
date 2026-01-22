import sys
from typing import (
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.token import (
import re
from codecs import BOM_UTF8, lookup
from . import token
def tokenize_loop(readline: Callable[[], str], tokeneater: TokenEater) -> None:
    for token_info in generate_tokens(readline):
        tokeneater(*token_info)