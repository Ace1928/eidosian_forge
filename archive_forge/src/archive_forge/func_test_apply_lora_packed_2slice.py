import pytest
import torch
from vllm.lora.layers import _apply_lora, _apply_lora_packed_nslice
from .utils import DummyLoRAManager
@pytest.mark.parametrize('m', TENSOR_SIZES)
@pytest.mark.parametrize('n', TENSOR_SIZES)
@pytest.mark.parametrize('k', BATCH_SIZES)
@pytest.mark.parametrize('rank', RANKS)
@pytest.mark.parametrize('dtype', DTYPES)
def test_apply_lora_packed_2slice(m, n, k, rank, dtype) -> None:
    if m % 2 != 0:
        pytest.skip('m must be divisible by 2')
    if m // 2 not in TENSOR_SIZES:
        pytest.skip('m//2 must be in TENSOR_SIZES')
    manager = DummyLoRAManager()
    module_name = 'module'
    weight = torch.rand([m // 2, n], device='cuda', dtype=dtype)
    manager.init_random_lora(module_name + '1', weight, rank=rank)
    lora_1 = manager.get_module_lora(module_name + '1')
    manager.init_random_lora(module_name + '2', weight, rank=rank)
    lora_2 = manager.get_module_lora(module_name + '2')
    input = torch.rand(k, n, device='cuda', dtype=dtype)
    expected = torch.cat([input @ lora_1.lora_a @ lora_1.lora_b * lora_1.scaling, input @ lora_2.lora_a @ lora_2.lora_b * lora_2.scaling], dim=1)
    lora_a_stacks = [torch.zeros(8, 1, lora_1.lora_a.shape[1], lora_1.lora_a.shape[0], device='cuda', dtype=dtype) for i in range(2)]
    lora_b_stacks = [torch.zeros(8, 1, lora_1.lora_b.shape[1], lora_1.lora_b.shape[0], device='cuda', dtype=dtype) for i in range(2)]
    for i in range(lora_a_stacks[0].shape[0]):
        lora_a_stacks[0][i][0] = lora_1.lora_a.T
        lora_b_stacks[0][i][0] = (lora_1.lora_b * lora_1.scaling).T
        lora_a_stacks[1][i][0] = lora_2.lora_a.T
        lora_b_stacks[1][i][0] = (lora_2.lora_b * lora_2.scaling).T
    output = torch.zeros(k, m, device='cuda', dtype=dtype)
    _apply_lora_packed_nslice(input, lora_a_stacks, lora_b_stacks, torch.randint(0, lora_a_stacks[0].shape[0], (len(input),), device='cuda'), output, (m // 2, m // 2))
    rtol, atol = TOLERANCES[dtype]
    assert torch.allclose(expected, output, rtol=rtol, atol=atol)
    output[:] = 0
    _apply_lora_packed_nslice(input, lora_a_stacks, lora_b_stacks, torch.full((len(input),), -1, device='cuda'), output, (m // 2, m // 2))
    assert torch.allclose(torch.zeros_like(output), output)
    manager.reset_lora()