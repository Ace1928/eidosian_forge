from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
@torch.no_grad()
def update_step(self, group, p, gindex, pindex):
    state = self.state[p]
    grad = p.grad
    config = self.get_config(gindex, pindex, group)
    state['step'] += 1
    step = state['step']
    if config['percentile_clipping'] < 100:
        current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(grad, state['gnorm_vec'], step, config['percentile_clipping'])
    else:
        gnorm_scale = 1.0
    if state['state1'].dtype == torch.float:
        F.optimizer_update_32bit(self.optimizer_name, grad, p, state['state1'], config['betas'][0], config['eps'], step, config['lr'], None, config['betas'][1], config['weight_decay'], gnorm_scale, state['unorm_vec'] if config['max_unorm'] > 0.0 else None, max_unorm=config['max_unorm'], skip_zeros=config['skip_zeros'])
    elif state['state1'].dtype == torch.uint8 and (not config['block_wise']):
        F.optimizer_update_8bit(self.optimizer_name, grad, p, state['state1'], None, config['betas'][0], config['betas'][1], config['eps'], step, config['lr'], state['qmap1'], None, state['max1'], None, state['new_max1'], None, config['weight_decay'], gnorm_scale, state['unorm_vec'] if config['max_unorm'] > 0.0 else None, max_unorm=config['max_unorm'])
        state['max1'], state['new_max1'] = (state['new_max1'], state['max1'])
    elif state['state1'].dtype == torch.uint8 and config['block_wise']:
        F.optimizer_update_8bit_blockwise(self.optimizer_name, grad, p, state['state1'], None, config['betas'][0], config['betas'][1], config['eps'], step, config['lr'], state['qmap1'], None, state['absmax1'], None, config['weight_decay'], gnorm_scale=gnorm_scale, skip_zeros=config['skip_zeros'])