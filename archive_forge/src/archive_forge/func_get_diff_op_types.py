import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import onnx
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.case.utils import import_recursive
from onnx.onnx_pb import (
def get_diff_op_types():
    cwd_path = Path.cwd()
    subprocess.run(['git', 'fetch', 'origin', 'main:main'], cwd=cwd_path, capture_output=True, check=True)
    obtain_diff = subprocess.Popen(['git', 'diff', '--name-only', '--diff-filter=AM', 'origin/main', 'HEAD'], cwd=cwd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdoutput, _ = obtain_diff.communicate()
    diff_list = stdoutput.split()
    changed_op_types = []
    for file in diff_list:
        file_name = file.decode('utf-8')
        if file_name.startswith('onnx/backend/test/case/node/') and file_name.endswith('.py'):
            changed_op_types.append(file_name.split('/')[-1].replace('.py', ''))
    return changed_op_types