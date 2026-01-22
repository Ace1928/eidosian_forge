from pythran.conversion import to_ast
from pythran.interval import UNKNOWN_RANGE, bool_values
from pythran.types.signature import extract_combiner
from pythran.typing import Any, Union, Fun, Generator
import gast as ast
def isconst(self):
    return not any((isinstance(x, UpdateEffect) for x in self.argument_effects)) and (not self.global_effects)