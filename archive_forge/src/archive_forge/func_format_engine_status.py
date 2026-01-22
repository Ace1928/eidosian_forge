from time import time  # noqa: F401
from typing import TYPE_CHECKING, Any, List, Tuple
def format_engine_status(engine: 'ExecutionEngine') -> str:
    checks = get_engine_status(engine)
    s = 'Execution engine status\n\n'
    for test, result in checks:
        s += f'{test:<47} : {result}\n'
    s += '\n'
    return s