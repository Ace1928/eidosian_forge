import torch
import torch.utils.benchmark as benchmark
def pytorch_profiler(fn, *inputs, trace_filename=None, backward=False, amp=False, amp_dtype=torch.float16, cpu=False, verbose=True, **kwinputs):
    """Wrap benchmark functions in Pytorch profiler to see CUDA information."""
    if backward:
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
            g = torch.randn_like(out)
    for _ in range(30):
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        if backward:
            out.backward(g, retain_graph=True)
    activities = ([torch.profiler.ProfilerActivity.CPU] if cpu else []) + [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        if backward:
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    x.grad = None
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        if backward:
            out.backward(g, retain_graph=True)
    if verbose:
        print(prof.key_averages().table(row_limit=50))
    if trace_filename is not None:
        prof.export_chrome_trace(trace_filename)