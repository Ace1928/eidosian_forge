from typing import Any, Dict, Iterable, List, no_type_check, Type
import torch
def optimizer_hook(*_unused) -> None:
    for opt in param._in_backward_optimizers:
        opt.step()
    param.grad = None