import pytest
import torch
import vllm.lora.punica as punica
@pytest.mark.parametrize('dtype_str', ['float16', 'bfloat16'])
@pytest.mark.parametrize('h1', H1)
@pytest.mark.parametrize('h2', H2)
@pytest.mark.parametrize('seed', SEED)
@torch.inference_mode()
def test_lora_correctness(dtype_str, h1, h2, seed):
    torch.manual_seed(seed)
    num_loras = 4
    num_layers = 1
    r = 8
    bs = 32
    scale = 0.123
    dtype = getattr(torch, dtype_str)
    device = torch.device('cuda')
    wa_T_all = torch.randn(num_loras, num_layers, r, h1, dtype=dtype, device=device)
    wb_T_all = torch.randn(num_loras, num_layers, h2, r, dtype=dtype, device=device)
    indices = torch.randint(num_loras, (bs,), dtype=torch.long, device=device)
    for layer_idx in range(num_layers):
        x = torch.randn(bs, h1, dtype=dtype, device=device)
        y = torch.randn(bs, h2, dtype=dtype, device=device)
        y_ref = y.clone()
        _lora_ref_impl(y_ref, x, wa_T_all, wb_T_all, indices, layer_idx, scale)
        y_our = y.clone()
        punica.add_lora(y_our, x, wa_T_all, wb_T_all, indices, layer_idx, scale)
        assert_close(y_ref, y_our)