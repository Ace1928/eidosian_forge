from typing import Any, Dict
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components.reversible import ReversibleSequence
def bench_revnet(backward: bool):
    device = torch.device('cuda')
    bw = '+bw' if backward else ''
    for dtype in [torch.float16, torch.float32]:
        results: Dict[str, Any] = {}
        for B, K in SHAPES:
            for depth in DEPTH:
                f = torch.nn.Linear(K, K).to(device=device, dtype=dtype)
                g = torch.nn.Linear(K, K).to(device=device, dtype=dtype)
                revseq = ReversibleSequence(torch.nn.ModuleList([torch.nn.ModuleList([f, g])] * depth))
                revseq = revseq.to(device=device, dtype=dtype)
                a = torch.rand(1, B, K, device=device, dtype=dtype, requires_grad=backward)
                b = torch.rand(1, B, K * 2, device=device, dtype=dtype, requires_grad=backward)

                def normal_step():
                    y = a
                    for _ in range(depth):
                        y = y + f(y)
                        y = y + g(y)
                    if backward:
                        torch.norm(y).backward()
                    return y

                def reversible_step():
                    y = revseq(b)
                    if backward:
                        torch.norm(y).backward()
                    return y
                for testcase in [TestCase(normal_step, f'residual - fw{bw}'), TestCase(reversible_step, f'reversible - fw{bw}')]:
                    time = triton.testing.do_bench(testcase.function)[0]
                    key = f'Batch={B}, Features={K}, Depth={depth}'
                    if key not in results:
                        results[key] = {}
                    results[key][testcase.name] = f'{time:.2f}'
        pretty_print(results, title=f'\n --- Type: {dtype} --- ', units='runtime in ms, lower is better')
        pretty_plot(results, title=f'RevNet-FW{bw}-{dtype}', units='runtime in ms, lower is better', dash_key='torch')