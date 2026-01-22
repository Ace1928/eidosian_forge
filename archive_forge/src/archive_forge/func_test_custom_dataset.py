import pytest
from unittest.mock import patch
from transformers import LlamaTokenizer
@pytest.mark.skip_missing_tokenizer
@patch('llama_recipes.finetuning.train')
@patch('llama_recipes.finetuning.AutoTokenizer')
@patch('llama_recipes.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_recipes.finetuning.optim.AdamW')
@patch('llama_recipes.finetuning.StepLR')
def test_custom_dataset(step_lr, optimizer, get_model, tokenizer, train, mocker, setup_tokenizer, llama_version):
    from llama_recipes.finetuning import main
    setup_tokenizer(tokenizer)
    skip_special_tokens = llama_version == 'meta-llama/Llama-2-7b-hf'
    kwargs = {'dataset': 'custom_dataset', 'model_name': llama_version, 'custom_dataset.file': 'recipes/finetuning/datasets/custom_dataset.py', 'custom_dataset.train_split': 'validation', 'batch_size_training': 2, 'val_batch_size': 4, 'use_peft': False, 'batching_strategy': 'padding'}
    main(**kwargs)
    assert train.call_count == 1
    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    tokenizer = args[3]
    assert len(train_dataloader) == 1120
    assert len(eval_dataloader) == 1120 // 2
    it = iter(eval_dataloader)
    batch = next(it)
    STRING = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=skip_special_tokens)
    assert STRING.startswith(EXPECTED_RESULTS[llama_version]['example_1'])
    assert batch['input_ids'].size(0) == 4
    assert set(('labels', 'input_ids', 'attention_mask')) == set(batch.keys())
    check_padded_entry(batch, tokenizer)
    it = iter(train_dataloader)
    next(it)
    batch = next(it)
    STRING = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=skip_special_tokens)
    assert STRING.startswith(EXPECTED_RESULTS[llama_version]['example_2'])
    assert batch['input_ids'].size(0) == 2
    assert set(('labels', 'input_ids', 'attention_mask')) == set(batch.keys())
    check_padded_entry(batch, tokenizer)