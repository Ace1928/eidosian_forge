import argparse
from typing import Any, Dict
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation
from xformers.components.feedforward import MLP, FusedMLP
def bench_MLP(backward: bool, bias: bool, dropout: float, activation: Activation):
    device = torch.device('cuda')
    bw = '+bw' if backward else ''
    for dtype in [torch.float16, torch.float32]:
        results: Dict[str, Any] = {}
        for B, M, K in SHAPES:
            for hlm in HIDDEN_LAYER_MULTIPLIER:
                fused_mlp = FusedMLP(dim_model=K, dropout=dropout, activation=activation, hidden_layer_multiplier=hlm, bias=bias).to(device=device, dtype=dtype)
                standard_mlp = MLP(dim_model=K, dropout=dropout, activation=activation, hidden_layer_multiplier=hlm, bias=bias).to(device=device, dtype=dtype)
                a = torch.randn((B, M, K), requires_grad=backward, device=device, dtype=dtype)

                def mlp_standard():
                    y = standard_mlp(a)
                    if backward:
                        torch.norm(y).backward()
                    return y

                def mlp_fused():
                    y = fused_mlp(a)
                    if backward:
                        torch.norm(y).backward()
                    return y
                for testcase in [TestCase(mlp_standard, 'standard - {} - {} bias - {} drop - fw{}'.format(activation, 'no' if not bias else '', dropout, '+bw' if backward else '')), TestCase(mlp_fused, 'fused - {} - {} bias - {} drop - fw{}'.format(activation, 'no' if not bias else '', dropout, '+bw' if backward else ''))]:
                    time = triton.testing.do_bench(testcase.function)[0]
                    key = f'{B} x {M} x {K} - {hlm}'
                    if key not in results:
                        results[key] = {}
                    results[key][testcase.name] = f'{time:.2f}'
        pretty_print(results, title=f'\n --- Type: {dtype} --- ', units='runtime in ms, lower is better. BMK - mul: ')
        pretty_plot(results, title=f'MLP-{activation}-FW{bw}-{dtype}', units='runtime in ms, lower is better', dash_key='torch')