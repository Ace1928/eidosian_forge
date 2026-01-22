import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate.utils.dataclasses import DistributedType
def mocked_dataloaders(accelerator, batch_size: int=16):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    data_files = {'train': 'tests/test_samples/MRPC/train.csv', 'validation': 'tests/test_samples/MRPC/dev.csv'}
    datasets = load_dataset('csv', data_files=data_files)
    label_list = datasets['train'].unique('label')
    label_to_id = {v: i for i, v in enumerate(label_list)}

    def tokenize_function(examples):
        outputs = tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=None, padding='max_length')
        if 'label' in examples:
            outputs['labels'] = [label_to_id[l] for l in examples['label']]
        return outputs
    tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=['sentence1', 'sentence2', 'label'])

    def collate_fn(examples):
        if accelerator.distributed_type == DistributedType.XLA:
            return tokenizer.pad(examples, padding='max_length', max_length=128, return_tensors='pt')
        return tokenizer.pad(examples, padding='longest', return_tensors='pt')
    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, collate_fn=collate_fn, batch_size=2)
    eval_dataloader = DataLoader(tokenized_datasets['validation'], shuffle=False, collate_fn=collate_fn, batch_size=1)
    return (train_dataloader, eval_dataloader)