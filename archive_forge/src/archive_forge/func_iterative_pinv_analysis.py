from typing import Callable
import torch
from torch.utils import benchmark
from xformers.components.attention.utils import iterative_pinv
def iterative_pinv_analysis(identity_tolerance: float=0.1, pinv_tolerance: float=0.5, max_iters: int=30, plot: bool=True):
    for i in range(1, 10):
        B, M = (1, 2 ** i)
        a = torch.rand(B, M, M)
        a = torch.softmax(a, dim=-1)
        for n_iter in range(1, max_iters + 1):
            result = iterative_pinv(a, n_iter=n_iter)
            expected = torch.linalg.pinv(a)
            result_identity = torch.matmul(a, result)
            identity = torch.eye(M)
            identity_error = torch.linalg.norm(identity - result_identity, dim=(-2, -1))
            inverse_error = torch.linalg.norm(expected - result, dim=(-2, -1))
            if (identity_error < identity_tolerance).all() or n_iter == max_iters:
                print(f'Size {M}, n_iters {n_iter}: \n\t                     Final Error from Identity: {identity_error.item()} \n\t                     Final Error from linalg.pinv {inverse_error.item()}')
                break