import pytest
from pytest import approx
from unittest.mock import patch
import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
from llama_recipes.finetuning import main
from llama_recipes.data.sampler import LengthBasedBatchSampler
@patch('llama_recipes.finetuning.torch.cuda.is_available')
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.AutoTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.get_preprocessed_dataset')
@patch('llama_recipes.finetuning.generate_peft_config')
@patch('llama_recipes.finetuning.get_peft_model')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
@pytest.mark.parametrize('cuda_is_available', [True, False])
def test_finetuning_peft(step_lr, optimizer, get_peft_model, gen_peft_config, get_dataset, tokenizer, get_model, train, cuda, cuda_is_available):
    kwargs = {'use_peft': True}
    get_dataset.return_value = get_fake_dataset()
    cuda.return_value = cuda_is_available
    main(**kwargs)
    if cuda_is_available:
        assert get_peft_model.return_value.to.call_count == 1
        assert get_peft_model.return_value.to.call_args.args[0] == 'cuda'
    else:
        assert get_peft_model.return_value.to.call_count == 0
    assert get_peft_model.return_value.print_trainable_parameters.call_count == 1