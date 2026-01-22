import itertools
import torch
from torch.utils import benchmark
from xformers.components.attention._sputnik_sparse import _csr_to_coo
from xformers.components.attention.core import SparseCS, _create_random_sparsity
def bench_sddmm(configs):
    min_run_time = MIN_RUN_TIME
    device = torch.device('cuda')
    results = []
    for (B, M, K), prob in configs:
        a = torch.rand(B, M, K, device=device)
        b = torch.rand(B, M, K, device=device)
        mask = _create_random_sparsity(torch.ones(1, M, M, dtype=torch.bool), prob, divisible_by=16)
        aa = a
        bb = b
        mask = SparseCS(mask, device)
        row_indices = mask.row_indices
        row_offsets = mask.row_offsets
        column_indices = mask.column_indices
        for backend in ['csr_sputnik', 'csr_ge', 'coo_ge', 'csr_to_coo']:
            fn_str = 'fn(a, b, row_indices, row_offsets, column_indices)'
            fn = _get_fn(backend)
            results.append(benchmark.Timer(stmt=fn_str, globals={'a': aa, 'b': bb, 'mask': mask, 'row_indices': row_indices, 'row_offsets': row_offsets, 'column_indices': column_indices, 'fn': fn}, label='sddmm', sub_label=f'B={B:>4d}, M={M:>4d}, K={K:>3d}, prob={prob:0.4f}', description=backend).blocked_autorange(min_run_time=min_run_time))
    compare = benchmark.Compare(results)
    compare.print()
    return results