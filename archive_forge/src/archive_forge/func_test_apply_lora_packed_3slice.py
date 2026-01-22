import pytest
import torch
from vllm.lora.layers import _apply_lora, _apply_lora_packed_nslice
from .utils import DummyLoRAManager
@pytest.mark.parametrize('qkv', QKV_TENSOR_SIZES)
@pytest.mark.parametrize('n', TENSOR_SIZES)
@pytest.mark.parametrize('k', BATCH_SIZES)
@pytest.mark.parametrize('rank', RANKS)
@pytest.mark.parametrize('dtype', DTYPES)
def test_apply_lora_packed_3slice(qkv, n, k, rank, dtype) -> None:
    manager = DummyLoRAManager()
    module_name = 'module'
    weight_q = torch.empty(qkv[0], n, device='cuda', dtype=dtype)
    weight_kv = torch.empty(qkv[1], n, device='cuda', dtype=dtype)
    manager.init_random_lora(module_name + 'q', weight_q, rank=rank)
    lora_q = manager.get_module_lora(module_name + 'q')
    manager.init_random_lora(module_name + 'k', weight_kv, rank=rank)
    lora_k = manager.get_module_lora(module_name + 'k')
    manager.init_random_lora(module_name + 'v', weight_kv, rank=rank)
    lora_v = manager.get_module_lora(module_name + 'v')
    input = torch.rand(k, n, device='cuda', dtype=dtype)
    expected = torch.cat([input @ lora_q.lora_a @ lora_q.lora_b * lora_q.scaling, input @ lora_k.lora_a @ lora_k.lora_b * lora_k.scaling, input @ lora_v.lora_a @ lora_v.lora_b * lora_v.scaling], dim=1)
    lora_a_stacks = [torch.zeros(8, 1, lora_q.lora_a.shape[1], lora_q.lora_a.shape[0], device='cuda', dtype=dtype)] + [torch.zeros(8, 1, lora_k.lora_a.shape[1], lora_k.lora_a.shape[0], device='cuda', dtype=dtype) for i in range(2)]
    lora_b_stacks = [torch.zeros(8, 1, lora_q.lora_b.shape[1], lora_q.lora_b.shape[0], device='cuda', dtype=dtype)] + [torch.zeros(8, 1, lora_k.lora_b.shape[1], lora_k.lora_b.shape[0], device='cuda', dtype=dtype) for i in range(2)]
    for i in range(lora_a_stacks[0].shape[0]):
        lora_a_stacks[0][i][0] = lora_q.lora_a.T
        lora_b_stacks[0][i][0] = (lora_q.lora_b * lora_q.scaling).T
        lora_a_stacks[1][i][0] = lora_k.lora_a.T
        lora_b_stacks[1][i][0] = (lora_k.lora_b * lora_k.scaling).T
        lora_a_stacks[2][i][0] = lora_v.lora_a.T
        lora_b_stacks[2][i][0] = (lora_v.lora_b * lora_v.scaling).T
    output = torch.zeros(k, sum(qkv), device='cuda', dtype=dtype)
    _apply_lora_packed_nslice(input, lora_a_stacks, lora_b_stacks, torch.randint(0, lora_a_stacks[0].shape[0], (len(input),), device='cuda'), output, (qkv[0], qkv[1], qkv[2]))
    rtol, atol = TOLERANCES[dtype]
    assert torch.allclose(expected, output, rtol=rtol, atol=atol)
    output[:] = 0
    _apply_lora_packed_nslice(input, lora_a_stacks, lora_b_stacks, torch.full((len(input),), -1, device='cuda'), output, (qkv[0], qkv[1], qkv[2]))
    assert torch.allclose(torch.zeros_like(output), output)
    manager.reset_lora()