from typing import Any, Dict, List, Optional
import torch
import triton
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print
from xformers.components import Activation, build_activation
from xformers.triton.fused_linear_layer import FusedLinear
def bench_linear(activations: List[Optional[Activation]]):
    device = torch.device('cuda')
    for dtype in [torch.float32, torch.float16]:
        for backward in [True, False]:
            for activation in activations:
                results: Dict[str, Any] = {}
                for bias in [False, True]:
                    for B, M, K in SHAPES:
                        a = torch.rand(B, M, K, device=device, dtype=dtype, requires_grad=backward)
                        torch_linear = torch.nn.Linear(K, 4 * K, bias=bias).to(dtype=dtype, device=device)
                        torch_activation = build_activation(activation)
                        fused_linear = FusedLinear(K, 4 * K, bias=bias, activation=activation).to(dtype=dtype, device=device)

                        def torch_step(x):
                            y = torch_activation(torch_linear(x))
                            if backward:
                                torch.norm(y).backward()
                            return y

                        def triton_step(x):
                            y = fused_linear(x)
                            if backward:
                                torch.norm(y).backward()
                            return y
                        metrics_transform = get_metrics_transform(activation, a, torch_linear.weight, torch_linear.bias, backward)
                        for testcase in [TestCase(torch_step, 'pytorch - {} - {} bias - fw{}'.format(activation, 'no' if not bias else '', '+bw' if backward else '')), TestCase(triton_step, 'triton  - {} - {} bias - fw{}'.format(activation, 'no' if not bias else '', '+bw' if backward else ''))]:
                            time = triton.testing.do_bench(lambda: testcase.function(a))[0]
                            key = f'B={B}, M={M}, K={K}'
                            if key not in results:
                                results[key] = {}
                            metric = metrics_transform(time)
                            results[key][testcase.name] = f'{metric:.1f}'
                pretty_print(results, title='\n --- Type: {} ---'.format(dtype), units='TFlops/s')
                _type = '_fp16' if dtype == torch.float16 else '_fp32'
                title = 'FusedLinear' + _type + '_FW'
                if backward:
                    title += '_BW'
                title += '_' + activation.value if activation else '_none'
                pretty_plot(results, title, 'TFlops/s', dash_key='pytorch')