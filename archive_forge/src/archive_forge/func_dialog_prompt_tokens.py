import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
def dialog_prompt_tokens(tokenizer: Tokenizer, dialog: Dialog) -> List[int]:
    """
    Prompt formatting for multi-turn dialogs.
    The dialog is expected to start with a system message and then alternate
    between user and assistant messages.
    """
    assert tokenizer.step_id is not None
    assert all([msg['role'] == 'user' for msg in dialog[1::2]]) and all([msg['role'] == 'assistant' for msg in dialog[2::2]]), "model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    assert dialog[-1]['role'] == 'user', f'Last message must be from user, got {dialog[-1]['role']}'
    dialog_tokens: List[int] = [tokenizer.bos_id]
    headers: List[str] = []
    for message in dialog:
        headers.clear()
        headers.append(f'Source: {message['role'].strip()}')
        if message.get('destination') is not None:
            headers.append(f'Destination: {message['destination'].strip()}')
        header = ' ' + '\n'.join(headers)
        dialog_tokens += tokenizer.encode(header, bos=False, eos=False)
        if message['content']:
            body = '\n\n ' + message['content'].strip()
            dialog_tokens += tokenizer.encode(body, bos=False, eos=False)
        dialog_tokens += [tokenizer.step_id]
    headers.clear()
    headers.append('Source: assistant')
    headers.append('Destination: user')
    header = ' ' + '\n'.join(headers)
    dialog_tokens += tokenizer.encode(header, bos=False, eos=False)
    dialog_tokens += tokenizer.encode('\n\n ', bos=False, eos=False)
    return dialog_tokens