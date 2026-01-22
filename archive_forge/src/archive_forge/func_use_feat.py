import random
import io
import os
import pickle
from parlai.utils.misc import msg_to_str
def use_feat(opt, field, ttype):
    if 'all' in opt.get(field) or opt.get(field) == ttype:
        return True
    else:
        return False