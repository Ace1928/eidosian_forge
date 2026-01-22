import inspect
import re
import warnings
from typing import Any, Dict
import torch
from torch.testing import make_tensor
def optimize_bsr_dense_addmm(m, k, n, bm, bk, beta=1, alpha=1, dtype=torch.float16, device='cuda', sparsity=0.5, force=False):
    import triton
    from torch.sparse._triton_ops import bsr_dense_addmm
    key = (m, k, n, bm, bk, beta == 0, beta == 1, alpha == 1)
    version = (0, dtype, sparsity)
    reference_meta = dict(GROUP_SIZE_ROW=1, num_stages=1, num_warps=4, SPLIT_N=max(n // bm, 1))
    initial_meta = get_meta('bsr_dense_addmm', key, version=version, exact=True)
    if initial_meta is None:
        initial_meta = get_meta('bsr_dense_addmm', key, version=(0, dtype, 0.5), exact=True)
        if initial_meta is None:
            initial_meta = reference_meta
    elif not force:
        return
    print(f'key, initial_meta={(key, initial_meta)!r}')
    torch.manual_seed(0)
    bsr = create_blocked_tensor(0, m, k, (bm, bk), sparsity, dtype, device).to_sparse_bsr((bm, bk))
    dense = make_tensor(k, n, dtype=dtype, device=device)
    input = make_tensor(m, n, dtype=dtype, device=device)

    def bench(meta, bsr=bsr, dense=dense):

        def test_func():
            return bsr_dense_addmm(input, bsr, dense, beta=beta, alpha=alpha, meta=meta)
        ms_min = triton.testing.do_bench(test_func, warmup=500, rep=100, fast_flush=False)
        return ms_min

    def step_meta_parameter(name, value, direction, meta, m=m, n=n, k=k, bm=bm, bk=bk):
        is_log = name in {'SPLIT_N', 'num_warps'}
        min_value = dict(SPLIT_N=1, num_warps=1, num_stages=1, GROUP_SIZE_ROW=1)[name]
        max_value = dict(SPLIT_N=max(n // bm, 1)).get(name)
        value_step = dict(SPLIT_N=2, num_warps=2, num_stages=1, GROUP_SIZE_ROW=1)[name]
        if is_log:
            next_value = value * value_step ** direction if direction > 0 else value // value_step ** abs(direction)
        else:
            next_value = value + value_step * direction
        if min_value is not None:
            next_value = max(next_value, min_value)
        if max_value is not None:
            next_value = min(next_value, max_value)
        if name == 'SPLIT_N' and n % next_value != 0:
            return value
        return next_value
    meta, speedup, timing, sensitivity_message = minimize(bench, initial_meta, reference_meta, step_meta_parameter, max_step=1)
    if initial_meta is not reference_meta and initial_meta == meta and (not force):
        return
    print(f'-> {sensitivity_message}, speedup={speedup:.1f} %, timing={timing:.3f} ms')
    if speedup < 0:
        return
    device_name = torch.cuda.get_device_name()
    update('bsr_dense_addmm', device_name, version, key, tuple((meta[k] for k in sorted(meta))))