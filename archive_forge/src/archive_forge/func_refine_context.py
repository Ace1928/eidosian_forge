import json, time, random, os
import numpy as np
import torch
from torch.nn import functional as F
def refine_context(self, context):
    context = context.strip().split('\n')
    for c in range(len(context)):
        context[c] = context[c].strip().strip('\u3000').strip('\r')
    context = list(filter(lambda c: c != '', context))
    context = '\n' + '\n'.join(context).strip()
    if context == '':
        context = '\n'
    return context