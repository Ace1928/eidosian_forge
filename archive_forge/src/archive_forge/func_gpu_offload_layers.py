import os
import multiprocessing
from typing import TypeVar, Optional, Tuple, List
def gpu_offload_layers(self, layer_count: int) -> bool:
    """
        Offloads specified count of model layers onto the GPU. Offloaded layers are evaluated using cuBLAS or CLBlast.
        For the purposes of this function, model head (unembedding matrix) is treated as an additional layer:
        - pass `model.n_layer` to offload all layers except model head
        - pass `model.n_layer + 1` to offload all layers, including model head

        Returns true if at least one layer was offloaded.
        If rwkv.cpp was compiled without cuBLAS and CLBlast support, this function is a no-op and always returns false.

        Parameters
        ----------
        layer_count : int
            Count of layers to offload onto the GPU, must be >= 0.
        """
    if not layer_count >= 0:
        raise ValueError('Layer count must be >= 0')
    return self._library.rwkv_gpu_offload_layers(self._ctx, layer_count)