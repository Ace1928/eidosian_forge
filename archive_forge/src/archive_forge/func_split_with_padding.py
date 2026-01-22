from typing import Union
import numpy as np
from onnx.reference.op_run import OpRun
def split_with_padding(x, separator=None, maxsplit=None):
    split_lists = np.char.split(x.astype(np.str_), separator, maxsplit)
    num_splits = np.vectorize(len, otypes=[np.int64])(split_lists)
    padding_requirement = (np.max(num_splits, initial=0) - num_splits).tolist()
    split_lists_padded = np.array(pad_empty_string(split_lists, padding_requirement), dtype=object)
    if x.size == 0:
        split_lists_padded = split_lists_padded.reshape(*x.shape, 0)
    return (split_lists_padded, num_splits)