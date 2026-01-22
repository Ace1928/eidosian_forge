import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def bucket_contvar(ex, ctrl, num_buckets):
    """
    Given ex, which contains a continuous value for a particular control variable,
    return the bucketed version of that control value.

    Inputs:
      ex: message dictionary. Assume it has key ctrl, mapping to the value.
      ctrl: string. The name of the CT control.
      num_buckets: int. The number of buckets for this control variable.
    """
    if ctrl not in ex.keys():
        raise ValueError('Control %s not found in example. Available keys in this example: %s' % (ctrl, ', '.join(ex.keys())))
    ctrl_val = ex[ctrl]
    if ctrl == 'avg_nidf':
        ctrl_val = float(ctrl_val)
        assert ctrl_val >= 0
        assert ctrl_val <= 1
    elif ctrl == 'lastuttsim':
        if ctrl_val == 'None':
            assert num_buckets == 11
            return 10
        else:
            ctrl_val = float(ctrl_val)
            assert ctrl_val >= -1
            assert ctrl_val <= 1
    else:
        raise ValueError('Unexpected CT ctrl: %s' % ctrl)
    bucket_lbs = CONTROL2BUCKETLBS[ctrl, num_buckets]
    if ctrl == 'lastuttsim':
        assert len(bucket_lbs) == num_buckets - 1
    else:
        assert len(bucket_lbs) == num_buckets
    return sort_into_bucket(ctrl_val, bucket_lbs)