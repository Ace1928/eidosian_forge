import datetime
import inspect
import re
import statistics
from functools import wraps
from sqlglot import exp
from sqlglot.generator import Generator
from sqlglot.helper import PYTHON_VERSION, is_int, seq_get
@null_if_any('this', 'expression')
def jsonextract(this, expression):
    for path_segment in expression:
        if isinstance(this, dict):
            this = this.get(path_segment)
        elif isinstance(this, list) and is_int(path_segment):
            this = seq_get(this, int(path_segment))
        else:
            raise NotImplementedError(f'Unable to extract value for {this} at {path_segment}.')
        if this is None:
            break
    return this