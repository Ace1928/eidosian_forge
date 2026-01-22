from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def wrapped_merge_func(op1, op2):
    wrapped_merge_func.num_function_calls += 1
    return merge_func(op1, op2)