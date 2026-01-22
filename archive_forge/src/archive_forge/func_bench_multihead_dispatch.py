from typing import Any, Dict
import torch
import torch.nn as nn
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import MultiHeadDispatch
from xformers.components.attention import ScaledDotProduct
def bench_multihead_dispatch(backward: bool, self_attention: bool):
    device = torch.device('cuda')
    bw = '+bw' if backward else ''
    sa = ' (self_attn)' if self_attention else ''
    for dtype in [torch.float16, torch.float32]:
        results: Dict[str, Any] = {}
        for B, M, K in SHAPES:
            for heads in N_HEADS:
                xf_multi_head = MultiHeadDispatch(dim_model=K, residual_dropout=0.0, num_heads=heads, attention=ScaledDotProduct(), bias=(True, True, True, True)).to(device=device, dtype=dtype)
                torch_multi_head = nn.MultiheadAttention(embed_dim=K, num_heads=heads, batch_first=True).to(device=device, dtype=dtype)
                q = torch.randn((B, M, K), requires_grad=backward, device=device, dtype=dtype)
                if self_attention:
                    k = q
                    v = q
                else:
                    k = torch.randn((B, M, K), requires_grad=backward, device=device, dtype=dtype)
                    v = torch.randn((B, M, K), requires_grad=backward, device=device, dtype=dtype)

                def torch_mha():
                    y, _ = torch_multi_head(query=q, key=k, value=v)
                    if backward:
                        torch.norm(y).backward()
                    return y

                def xformers_mha():
                    y = xf_multi_head(query=q, key=k, value=v)
                    if backward:
                        torch.norm(y).backward()
                    return y
                for testcase in [TestCase(torch_mha, f'torch - fw{bw}{sa}'), TestCase(xformers_mha, f'xf - fw{bw}{sa}')]:
                    time = triton.testing.do_bench(testcase.function)[0]
                    key = f'B={B}, M={M}, K={K}, N_HEADS={heads}'
                    if key not in results:
                        results[key] = {}
                    results[key][testcase.name] = f'{time:.2f}'
        pretty_print(results, title=f'\n --- Type: {dtype} --- ', units='runtime in ms, lower is better')
        pretty_plot(results, title=f'MHA-FW{bw}-{dtype}', units='runtime in ms, lower is better', dash_key='torch')