import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def filter_unpredictable_numbers(text):
    return re.sub('u3\\(.+\\)', 'u3(<not-repeatable>)', text)