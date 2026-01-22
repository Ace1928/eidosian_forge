import dataclasses
import tempfile
from collections import defaultdict
import torch
from torch.autograd import DeviceType
from .utils import create_bandwidth_info_str, do_bench, get_num_bytes
def benchmark_all_kernels(benchmark_name, benchmark_all_configs):
    """
    An experimental API used only when config.benchmark_kernel is true.

    Run the kernel benchmarks for all the kernels cached in PyCodeCache.
    Used in the compiled modules.

    Put this method here rather than codegen it for convenience since its implementation
    does not change based on different graph modules being compiled.
    """
    from torch._inductor.codecache import PyCodeCache

    def get_triton_kernel(mod):
        from torch._inductor.triton_heuristics import CachingAutotuner
        cand_list = [v for k, v in mod.__dict__.items() if k.startswith('triton_') and isinstance(v, CachingAutotuner)]
        assert len(cand_list) == 1
        return cand_list[0]
    nfound = 0
    for kernel_key, kernel_mod in PyCodeCache.cache.items():
        if not hasattr(kernel_mod, 'get_args') or not hasattr(kernel_mod, 'call'):
            continue
        triton_kernel = get_triton_kernel(kernel_mod)
        kernel_category = get_kernel_category(kernel_mod)
        args = kernel_mod.get_args()
        num_in_out_ptrs = len([arg_name for arg_name in triton_kernel.fn.arg_names if arg_name.startswith('in_out_ptr')])
        num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1000000000.0

        def get_info_str(ms, n_regs, n_spills, shared, prefix=''):
            if not any((x is None for x in [n_regs, n_spills, shared])):
                kernel_detail_str = f'  {n_regs:3} regs  {n_spills:3} spills  {shared:8} shared mem'
            else:
                kernel_detail_str = ''
            gb_per_s = num_gb / (ms / 1000.0)
            return create_bandwidth_info_str(ms, num_gb, gb_per_s, prefix=prefix, suffix=kernel_detail_str)
        kernel_desc = f'{benchmark_name:20} {kernel_category[:3].upper()} {kernel_key[:10]}'
        if benchmark_all_configs:
            assert hasattr(kernel_mod, 'benchmark_all_configs')
            bench_result = kernel_mod.benchmark_all_configs(args)
            print(kernel_desc)
            for launcher, ms in bench_result.items():
                print(f'  {get_info_str(ms, launcher.n_regs, launcher.n_spills, launcher.shared)} @ {launcher.config}')
        else:
            ms = do_bench(lambda: kernel_mod.call(args), rep=40, fast_flush=True)
            assert len(triton_kernel.launchers) == 1, 'Autotuner should have selected the best config'
            launcher = triton_kernel.launchers[0]
            print(get_info_str(ms, launcher.n_regs, launcher.n_spills, launcher.shared, prefix=f'{kernel_desc} '))
        nfound += 1
    if nfound == 0:
        print('No kernel with benchmark functionality found. Make sure you run inductor with config.benchmark_kernel being True')