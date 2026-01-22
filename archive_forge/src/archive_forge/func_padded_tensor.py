import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
def padded_tensor(items: List[Union[List[int], torch.LongTensor]], pad_idx: int=0, use_cuda: bool=False, left_padded: bool=False, max_len: Optional[int]=None, fp16friendly: bool=False, device: int=-1) -> Tuple[torch.LongTensor, List[int]]:
    """
    Create a padded matrix from an uneven list of lists.

    Returns (padded, lengths), where padded is the padded matrix, and lengths
    is a list containing the lengths of each row.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param bool sort: If True, orders by the length
    :param int pad_idx: the value to use for padding
    :param bool use_cuda: if true, places `padded` on GPU
    :param bool left_padded:
    :param int max_len: if None, the max length is the maximum item length
    :param bool fp16friendly: if True, pads the time dimension to be a multiple of 4.
    :param int device: GPU device.

    :returns: (padded, lengths) tuple
    :rtype: (Tensor[int64], list[int])
    """
    n = len(items)
    lens: List[int] = [len(item) for item in items]
    t = max(lens) if max_len is None else max_len
    t = max(t, 1)
    if fp16friendly and t % FP16_PAD_SIZE != 0:
        t += FP16_PAD_SIZE - t % FP16_PAD_SIZE
    if isinstance(items[0], torch.Tensor):
        output = items[0].new(n, t)
    else:
        output = torch.LongTensor(n, t)
    output.fill_(pad_idx)
    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.LongTensor(item)
        if left_padded:
            output[i, t - length:] = item
        else:
            output[i, :length] = item
    if use_cuda:
        output = output.cuda()
        if device >= 0:
            output = output.to(device)
    return (output, lens)