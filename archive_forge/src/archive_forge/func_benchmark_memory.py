import torch
import torch.utils.benchmark as benchmark
def benchmark_memory(fn, *inputs, desc='', verbose=True, **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*inputs, **kwinputs)
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / (2 ** 20 * 1000)
    if verbose:
        print(f'{desc} max memory: {mem}GB')
    torch.cuda.empty_cache()
    return mem