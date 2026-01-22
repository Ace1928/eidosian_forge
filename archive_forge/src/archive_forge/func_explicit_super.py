import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple
from .bytecode_analysis import (
def explicit_super(code: types.CodeType, instructions: List[Instruction]) -> None:
    """convert super() with no args into explicit arg form"""
    cell_and_free = (code.co_cellvars or tuple()) + (code.co_freevars or tuple())
    output = []
    for idx, inst in enumerate(instructions):
        output.append(inst)
        if inst.opname == 'LOAD_GLOBAL' and inst.argval == 'super':
            nexti = instructions[idx + 1]
            if nexti.opname in ('CALL_FUNCTION', 'PRECALL') and nexti.arg == 0:
                assert '__class__' in cell_and_free
                output.append(create_instruction('LOAD_DEREF', argval='__class__'))
                first_var = code.co_varnames[0]
                if first_var in cell_and_free:
                    output.append(create_instruction('LOAD_DEREF', argval=first_var))
                else:
                    output.append(create_instruction('LOAD_FAST', argval=first_var))
                nexti.arg = 2
                nexti.argval = 2
                if nexti.opname == 'PRECALL':
                    call_inst = instructions[idx + 2]
                    call_inst.arg = 2
                    call_inst.argval = 2
    instructions[:] = output