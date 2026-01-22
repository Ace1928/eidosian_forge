from typing import Callable
import torch
from torch.utils import benchmark
from xformers.components.attention.utils import iterative_pinv
def bench_inverse(inverse_fn: Callable[[torch.Tensor], torch.Tensor]):
    min_run_time = MIN_RUN_TIME
    prob = 0.9
    device = torch.device('cuda')
    results = []
    for B, M, K in zip(*SHAPES):
        a = torch.rand(B, M, M, device=device)
        a[a < prob] = 0
        a = torch.softmax(a, dim=-1)
        results.extend([benchmark.Timer(stmt=f'{inverse_fn.__name__}(a)', globals={'a': a, f'{inverse_fn.__name__}': inverse_fn}, label=f'{inverse_fn.__name__}', sub_label='dense', description=f'B={B}, M={M}, K={K}').blocked_autorange(min_run_time=min_run_time)])
        for prob in SPARSITIES:
            a = torch.rand(B, M, M, device=device)
            a[a < prob] = 0
            a = a.to_sparse()
            results.append(benchmark.Timer(stmt=f'{inverse_fn.__name__}(a)', globals={'a': a, f'{inverse_fn.__name__}': inverse_fn}, label=f'{inverse_fn.__name__}', sub_label=f'sparsity: {prob:0.2f}', description=f'B={B}, M={M}, K={K}').blocked_autorange(min_run_time=min_run_time))
    compare = benchmark.Compare(results)
    compare.print()