import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_get_n_layer(self, ctx: RWKVContext) -> int:
    """
        Returns the number of layers in the given model.
        A layer is a pair of RWKV and FFN operations, stacked multiple times throughout the model.
        Embedding matrix and model head (unembedding matrix) are NOT counted in `n_layer`.
        Useful for always offloading the entire model to GPU.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        """
    return self.library.rwkv_get_n_layer(ctx.ptr)