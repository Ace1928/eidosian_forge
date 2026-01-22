from typing import Any, Dict, List, Union
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel
import gc
import copy
import global_var
def longest_prefix_state(body: LongestPrefixStateBody, request: Request):
    global trie
    if trie is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'trie not loaded')
    import torch
    import numpy as np
    id = -1
    try:
        for id, len in trie.prefix(body.prompt):
            pass
    except:
        pass
    if id != -1:
        prompt: str = trie[id]
        v = dtrie[id]
        tokens: List[Union[str, int]] = copy.deepcopy(v['tokens'])
        devices: List[torch.device] = v['devices']
        logits_device: Union[torch.device, None] = v['logits_device']
        state: Union[Any, None] = v['state']
        logits: Union[Any, None] = v['logits']
        if type(state) == list and hasattr(state[0], 'device'):
            state = [tensor.to(devices[i]) if devices[i] != torch.device('cpu') else tensor.clone() for i, tensor in enumerate(state)]
            logits = logits.to(logits_device) if logits_device != torch.device('cpu') else logits.clone()
        else:
            logits = np.copy(logits)
        quick_log(request, body, 'Hit:\n' + prompt)
        return {'prompt': prompt, 'tokens': tokens, 'state': state, 'logits': logits}
    else:
        return {'prompt': '', 'tokens': [], 'state': None, 'logits': None}