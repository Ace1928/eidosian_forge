import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
def get_batch_func(self, megatron_dataset_flag):

    def get_batch_megatron(data_iterator):
        """Build the batch."""
        keys = ['text_enc', 'text_dec', 'labels', 'loss_mask', 'enc_mask', 'dec_mask', 'enc_dec_mask']
        datatype = torch.int64
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        data_b = mpu.broadcast_data(keys, data, datatype)
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()
        enc_mask = data_b['enc_mask'] < 0.5
        dec_mask = data_b['dec_mask'] < 0.5
        enc_dec_mask = data_b['enc_dec_mask'] < 0.5
        return (tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask)

    def get_batch_transformer(data_iterator):
        """Build the batch."""
        data = next(data_iterator)
        data = send_to_device(data, torch.cuda.current_device())
        tokens_enc = data['input_ids'].long()
        labels = data['labels'].long()
        loss_mask = (labels != -100).to(torch.float)
        if 'decoder_input_ids' in data:
            tokens_dec = data['decoder_input_ids'].long()
        else:
            tokens_dec = labels.new_zeros(labels.shape, device=labels.device, dtype=torch.long)
            tokens_dec[..., 1:] = labels[..., :-1].clone()
            tokens_dec[..., 0] = 0
            tokens_dec.masked_fill_(tokens_dec == -100, 0)
        enc_mask = T5TrainStep.attn_mask_postprocess(data['attention_mask'].long())
        dec_mask = T5TrainStep.get_decoder_mask(tokens_dec.shape[1], tokens_dec.device)
        enc_dec_mask = T5TrainStep.get_enc_dec_mask(data['attention_mask'].long(), tokens_dec.shape[1], tokens_dec.device)
        return (tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, enc_dec_mask)
    if megatron_dataset_flag:
        return get_batch_megatron
    else:
        return get_batch_transformer