from typing import Optional
from dataclasses import dataclass
import argparse
import json
import os
import random
import numpy as np
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from ray.util.multiprocessing import Pool
def generate_split(conversations: list, tokenizer: transformers.AutoTokenizer, split_name: str, out_dir: str):

    def _convert_single_conversation(c):
        tokens = []
        masks = []
        if CONFIG.bos_token:
            t = tokenizer.convert_tokens_to_ids(CONFIG.bos_token)
            tokens.append(t)
            masks.append(False)
        if CONFIG.system:
            t = tokenizer(CONFIG.system, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids(CONFIG.eot_token)]
            tokens.extend(t)
            masks.extend([False] * len(t))
        for message in c['items']:
            message_text = CONFIG.role_prefix[message['from']] + message['value']
            t = tokenizer(message_text, add_special_tokens=False) + [tokenizer.convert_tokens_to_ids(CONFIG.eot_token)]
            tokens.extend(t)
            masks.extend([message['from'] == CONFIG.ai_role] * len(t))
        return (tokens, masks)
    converted = Pool().map(_convert_single_conversation, conversations)
    pad_id = tokenizer.convert_tokens_to_ids(CONFIG.pad_token)
    all_input_ids = []
    all_labels = []
    all_attention_masks = []
    all_plain_texts = []
    for tokens, masks in converted:
        tokens = np.array(tokens[:CONFIG.max_tokens], np.int_)
        masks = np.array(masks[:CONFIG.max_tokens], np.bool_)
        input_ids = np.full(CONFIG.max_tokens, pad_id, np.int_)
        labels = np.full(CONFIG.max_tokens, CONFIG.ignore_id, np.int_)
        attention_masks = np.full(CONFIG.max_tokens, False, np.bool_)
        length = len(tokens)
        input_ids[:length] = tokens
        labels[:length] = np.where(masks, tokens, CONFIG.ignore_id)
        attention_masks[:length] = True
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_masks.append(attention_masks)
        all_plain_texts.append(tokens)
    np.savez(os.path.join(out_dir, f'ochat.{split_name}.npz'), input_ids=np.vstack(all_input_ids), labels=np.vstack(all_labels), attention_masks=np.vstack(all_attention_masks))
    all_plain_texts = tokenizer.decode(all_plain_texts)
    with open(os.path.join(out_dir, f'ochat.{split_name}.text.json'), 'w') as f:
        json.dump(all_plain_texts, f)