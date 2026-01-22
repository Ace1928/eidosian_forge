import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
def hierarchy_distance(cls_combination):
    candidate_distance = sum((c[0] for c in cls_combination))
    if tuple((c[1] for c in cls_combination)) in registry:
        return candidate_distance
    return 10000