import contextlib
from typing import List, Tuple
import torch
@contextlib.contextmanager
def optimized_execution(should_optimize):
    """Context manager that controls whether the JIT's executor will run optimizations before executing a function."""
    stored_flag = torch._C._get_graph_executor_optimize()
    torch._C._set_graph_executor_optimize(should_optimize)
    try:
        yield
    finally:
        torch._C._set_graph_executor_optimize(stored_flag)