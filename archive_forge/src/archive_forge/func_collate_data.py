import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from datasets import load_dataset
def collate_data(blocks: List[Dict[str, torch.LongTensor]], contain_labels: bool=False, pad_token_id: Optional[int]=None) -> Dict[str, torch.LongTensor]:
    """
        Collate data in `blocks`
    Args:
        blocks (`List[Dict[str, torch.LongTensor]]`):
            List of tensors that we need to batch together
        pad_token_id (`Optional[int]`, defaults to `None`):
            Pad token id of the model
        contain_labels (`bool`, defaults to `False`):
           Set True to also process the labels

    Returns:
        `Dict[str, torch.LongTensor]`: Batched data
    """

    def pad_block(block, pads):
        return torch.cat((pads.to(block.device), block), dim=-1).long()
    input_ids_blocks = [block['input_ids'] for block in blocks]
    attention_mask_blocks = [block['attention_mask'] for block in blocks]
    if contain_labels:
        label_blocks = [block['labels'] for block in blocks]
        label_max_len = max([block.size(-1) for block in label_blocks])
    bsz = len(blocks)
    inp_max_len = max([block.size(-1) for block in input_ids_blocks])
    for i in range(bsz):
        block_bsz, block_inp_len = input_ids_blocks[i].shape
        pad_num = inp_max_len - block_inp_len
        if pad_num > 0:
            input_ids_blocks[i] = pad_block(input_ids_blocks[i], torch.ones((block_bsz, pad_num)) * pad_token_id)
            attention_mask_blocks[i] = pad_block(attention_mask_blocks[i], torch.zeros((block_bsz, pad_num)))
        if contain_labels:
            block_label_len = label_blocks[i].shape[-1]
            label_pad_num = label_max_len - block_label_len
            if label_pad_num > 0:
                label_blocks[i] = pad_block(label_blocks[i], torch.ones((block_bsz, label_pad_num)) * -100)
    data = {'input_ids': torch.cat(input_ids_blocks, dim=0).long(), 'attention_mask': torch.cat(attention_mask_blocks, dim=0).long()}
    if contain_labels:
        data['labels'] = torch.cat(label_blocks, dim=0).long()
    return data