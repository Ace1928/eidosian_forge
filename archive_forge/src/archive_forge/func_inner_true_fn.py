import torch
from torch._export.db.case import export_case
from functorch.experimental.control_flow import cond
def inner_true_fn(y):
    return x + y