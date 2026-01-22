import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
def log_tensor_stats(self, tensor, name):
    """Add distribution statistics on a tensor's elements to the current History entry"""
    if isinstance(tensor, (tuple, list)):
        while isinstance(tensor, (tuple, list)) and isinstance(tensor[0], (tuple, list)):
            tensor = [item for sublist in tensor for item in sublist]
        tensor = torch.cat([t.detach().clone().reshape(-1) for t in tensor])
    tensor = tensor.detach().clone()
    if not hasattr(tensor, 'shape'):
        cls = type(tensor)
        raise TypeError(f'Expected Tensor, not {cls.__module__}.{cls.__name__}')
    sparse_zeros = None
    if tensor.is_sparse:
        tensor = tensor.cpu().coalesce()
        backing_values = tensor._values()
        sparse_zeros = tensor.numel() - backing_values.numel()
        tensor = backing_values
    flat = tensor.reshape(-1)
    if flat.is_cuda:
        if self._is_cuda_histc_supported is None:
            try:
                flat.histc(bins=self._num_bins)
            except RuntimeError:
                self._is_cuda_histc_supported = False
            else:
                self._is_cuda_histc_supported = True
        if not self._is_cuda_histc_supported:
            flat = flat.cpu()
        elif not isinstance(flat, (torch.cuda.FloatTensor, torch.cuda.DoubleTensor)):
            flat = flat.type(torch.cuda.FloatTensor)
    if not flat.is_cuda and (not isinstance(flat, (torch.FloatTensor, torch.DoubleTensor))):
        flat = flat.type(torch.FloatTensor)
    if self._no_finite_values(flat):
        return
    flat = self._remove_infs_nans(flat)
    tmin = flat.min().item()
    tmax = flat.max().item()
    if sparse_zeros:
        tmin = 0 if tmin > 0 else tmin
        tmax = 0 if tmax < 0 else tmax
    if tmin > tmax:
        tmin, tmax = (tmax, tmin)
    if tmin == tmax:
        tensor = torch.Tensor([flat.numel()])
        tensor = tensor.cpu().clone().detach()
        bins = torch.Tensor([tmin, tmax])
    else:
        tensor = flat.histc(bins=self._num_bins, min=tmin, max=tmax)
        tensor = tensor.cpu().detach().clone()
        bins = torch.linspace(tmin, tmax, steps=self._num_bins + 1)
    if sparse_zeros:
        bins_np = bins.numpy()
        tensor_np = tensor.numpy()
        bin_idx = 0
        num_buckets = len(bins_np) - 1
        for i in range(num_buckets):
            start = bins_np[i]
            end = bins_np[i + 1]
            if start <= 0 and end > 0 or (i == num_buckets - 1 and end == 0):
                bin_idx = i
                break
        tensor_np[bin_idx] += sparse_zeros
        tensor = torch.Tensor(tensor_np)
        bins = torch.Tensor(bins_np)
    wandb.run._log({name: wandb.Histogram(np_histogram=(tensor.tolist(), bins.tolist()))}, commit=False)