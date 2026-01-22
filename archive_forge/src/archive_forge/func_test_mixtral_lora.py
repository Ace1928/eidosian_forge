import pytest
import torch
import vllm
from vllm.lora.request import LoRARequest
@pytest.mark.parametrize('tp_size', [4])
def test_mixtral_lora(mixtral_lora_files, tp_size):
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f'Not enough GPUs for tensor parallelism {tp_size}')
    llm = vllm.LLM(MODEL_PATH, enable_lora=True, max_num_seqs=16, max_loras=4, tensor_parallel_size=tp_size, worker_use_ray=True)
    expected_lora_output = ['give_opinion(name[SpellForce 3], release_year[2017], developer[Grimlore Games], rating[poor])', 'give_opinion(name[SpellForce 3], release_year[2017], developer[Grimlore Games], rating[poor])', 'inform(name[BioShock], release_year[2007], rating[good], genres[action-adventure, role-playing, shooter], platforms[PlayStation, Xbox, PC], available_on_steam[yes], has_linux_release[no], has_mac_release[yes])']
    assert do_sample(llm, mixtral_lora_files, lora_id=1) == expected_lora_output
    assert do_sample(llm, mixtral_lora_files, lora_id=2) == expected_lora_output