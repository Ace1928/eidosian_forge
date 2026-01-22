from NumPy.
import os
from pathlib import Path
import ast
import tokenize
import scipy
import pytest
@pytest.fixture(scope='session')
def warning_calls():
    base = Path(scipy.__file__).parent
    bad_filters = []
    bad_stacklevels = []
    for path in base.rglob('*.py'):
        with tokenize.open(str(path)) as file:
            tree = ast.parse(file.read(), filename=str(path))
            finder = FindFuncs(path.relative_to(base))
            finder.visit(tree)
            bad_filters.extend(finder.bad_filters)
            bad_stacklevels.extend(finder.bad_stacklevels)
    return (bad_filters, bad_stacklevels)