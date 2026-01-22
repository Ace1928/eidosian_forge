import json
import re
import tempfile
from typing import Any, List, Optional
from click import echo, style
from mypy_extensions import mypyc_attr
def ipynb_diff(a: str, b: str, a_name: str, b_name: str) -> str:
    """Return a unified diff string between each cell in notebooks `a` and `b`."""
    a_nb = json.loads(a)
    b_nb = json.loads(b)
    diff_lines = [diff(''.join(a_nb['cells'][cell_number]['source']) + '\n', ''.join(b_nb['cells'][cell_number]['source']) + '\n', f'{a_name}:cell_{cell_number}', f'{b_name}:cell_{cell_number}') for cell_number, cell in enumerate(a_nb['cells']) if cell['cell_type'] == 'code']
    return ''.join(diff_lines)