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
def test_shared_runtime_info():
    shared_rtinfo = cg.SharedRuntimeInfo(run_id='my run', run_start_time=datetime.datetime.now(tz=datetime.timezone.utc))
    cg_assert_equivalent_repr(shared_rtinfo)