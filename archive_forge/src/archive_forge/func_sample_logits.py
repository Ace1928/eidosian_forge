import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F
def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):
    lastChar = int(x[-1])
    probs = F.softmax(out, dim=-1)
    if self.charMode:
        if self.itos[lastChar] == '\n':
            top_p = top_p_newline
        else:
            top_p = top_p_usual
    else:
        top_p = top_p_usual
    if os.environ['RWKV_RUN_DEVICE'] == 'cpu':
        probs = probs.numpy()
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
        probs = probs / np.sum(probs)
        out = np.random.choice(a=len(probs), p=probs)
        return out
    else:
        sorted_probs = torch.sort(probs, descending=True)[0]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
        probs[probs < cutoff] = 0
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)
        out = torch.multinomial(probs, num_samples=1)[0]
        return out