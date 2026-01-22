import functools
import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.cpp_extension import load
def generate_init_weight(self):
    print(f'\n############################################################################\n#\n# Init model weight (slow for large models)...\n#\n############################################################################\n')
    m = {}
    for n in self.state_dict():
        p = self.state_dict()[n]
        shape = p.shape
        gain = 1.0
        scale = 1.0
        if 'ln_' in n or '.ln' in n or 'time_' in n or ('_mask' in n) or ('pos_emb' in n) or ('.mask.' in n):
            if 'ln_x.weight' in n:
                layer_scale = (1 + int(n.split('.')[1])) / self.args.n_layer
                m[n] = p * 0.0 + layer_scale ** 0.7
            else:
                m[n] = p
        else:
            if n == 'emb.weight':
                scale = -1 * self.args.lr_init
            else:
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                zero = ['.att.output.', '.ffn.value.', '.ffn.receptance.', '.ffnPre.value.', '.ffnPre.receptance.', 'head_q.', '.oo.', '.rr.']
                for kk in zero:
                    if kk in n:
                        scale = 0
                if n == 'head.weight':
                    scale = 0.5
                if 'head_k.' in n:
                    scale = 0.1
                if 'head_q.' in n:
                    scale = 0
            print(f'{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}')
            if self.args.accelerator.upper() == 'GPU':
                m[n] = torch.empty((shape[0], shape[1]), device='cuda')
            else:
                m[n] = torch.empty((shape[0], shape[1]))
            if scale == 0:
                nn.init.zeros_(m[n])
            elif scale < 0:
                nn.init.uniform_(m[n], a=scale, b=-scale)
            else:
                nn.init.orthogonal_(m[n], gain=gain * scale)
        m[n] = m[n].cpu()
        if os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            m[n] = m[n].half()
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            m[n] = m[n].bfloat16()
    gc.collect()
    torch.cuda.empty_cache()
    return m