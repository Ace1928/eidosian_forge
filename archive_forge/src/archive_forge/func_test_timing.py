import datetime
import glob
import re
import time
import uuid
from typing import List, cast, Any
import numpy as np
import pytest
import cirq
import cirq_google as cg
from cirq_google.workflow.quantum_executable_test import _get_quantum_executables, _get_example_spec
from cirq_google.workflow.quantum_runtime import _time_into_runtime_info
def test_timing():
    rt = cg.RuntimeInfo(execution_index=0)
    with _time_into_runtime_info(rt, 'test_proc'):
        time.sleep(0.1)
    assert 'test_proc' in rt.timings_s
    assert rt.timings_s['test_proc'] > 0.05