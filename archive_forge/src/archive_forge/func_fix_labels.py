import random
import io
import os
import pickle
from parlai.utils.misc import msg_to_str
def fix_labels(act, opt):
    labels = act.get('labels', act.get('eval_labels'))
    clip = int(opt.get('light_use_clip_cands', 1000))
    while len(act['label_candidates']) >= clip:
        act['label_candidates'].pop()
    is_label_cand = {}
    is_label_cand[labels] = False
    for c in act['label_candidates']:
        if c in is_label_cand:
            is_label_cand[c] = True
    for l, has in is_label_cand.items():
        if has is False:
            print('***ADDING A LABEL CAND****')
            act['label_candidates'].append(l)