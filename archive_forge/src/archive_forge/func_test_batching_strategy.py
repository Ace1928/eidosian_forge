import pytest
from pytest import approx
from unittest.mock import patch
import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler
from llama_recipes.finetuning import main
from llama_recipes.data.sampler import LengthBasedBatchSampler
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.AutoTokenizer.from_pretrained')
@patch('llama_recipes.finetuning.get_preprocessed_dataset')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_batching_strategy(step_lr, optimizer, get_dataset, tokenizer, get_model, train):
    kwargs = {'batching_strategy': 'packing'}
    get_dataset.return_value = get_fake_dataset()
    main(**kwargs)
    assert train.call_count == 1
    args, kwargs = train.call_args
    train_dataloader, eval_dataloader = args[1:3]
    assert isinstance(train_dataloader.batch_sampler, BatchSampler)
    assert isinstance(eval_dataloader.batch_sampler, BatchSampler)
    kwargs['batching_strategy'] = 'padding'
    train.reset_mock()
    main(**kwargs)
    assert train.call_count == 1
    args, kwargs = train.call_args
    train_dataloader, eval_dataloader = args[1:3]
    assert isinstance(train_dataloader.batch_sampler, LengthBasedBatchSampler)
    assert isinstance(eval_dataloader.batch_sampler, LengthBasedBatchSampler)
    kwargs['batching_strategy'] = 'none'
    with pytest.raises(ValueError):
        main(**kwargs)