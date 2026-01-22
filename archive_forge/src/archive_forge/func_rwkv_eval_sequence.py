import os
import sys
import ctypes
import pathlib
import platform
from typing import Optional, List, Tuple, Callable
def rwkv_eval_sequence(self, ctx: RWKVContext, tokens: List[int], state_in_address: Optional[int], state_out_address: int, logits_out_address: int) -> None:
    """
        Evaluates the model for a sequence of tokens.
        Uses a faster algorithm than `rwkv_eval` if you do not need the state and logits for every token. Best used with sequence lengths of 64 or so.
        Has to build a computation graph on the first call for a given sequence, but will use this cached graph for subsequent calls of the same sequence length.

        NOTE ON GGML NODE LIMIT

        ggml has a hard-coded limit on max amount of nodes in a computation graph. The sequence graph is built in a way that quickly exceedes
        this limit when using large models and/or large sequence lengths.
        Fortunately, rwkv.cpp's fork of ggml has increased limit which was tested to work for sequence lengths up to 64 for 14B models.

        If you get `GGML_ASSERT: ...\\ggml.c:16941: cgraph->n_nodes < GGML_MAX_NODES`, this means you've exceeded the limit.
        To get rid of the assertion failure, reduce the model size and/or sequence length.

        Not thread-safe. For parallel inference, call `rwkv_clone_context` to create one rwkv_context for each thread.
        Throws an exception in case of any error. Error messages would be printed to stderr.

        Parameters
        ----------
        ctx : RWKVContext
            RWKV context obtained from rwkv_init_from_file.
        tokens : List[int]
            Next token indices, in range 0 <= token < n_vocab.
        state_in_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count; or None, if this is a first pass.
        state_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_state_buffer_element_count. This buffer will be written to.
        logits_out_address : int
            Address of the first element of a FP32 buffer of size rwkv_get_logits_buffer_element_count. This buffer will be written to.
        """
    if not self.library.rwkv_eval_sequence(ctx.ptr, ctypes.cast((ctypes.c_int32 * len(tokens))(*tokens), P_INT), ctypes.c_size_t(len(tokens)), ctypes.cast(0 if state_in_address is None else state_in_address, P_FLOAT), ctypes.cast(state_out_address, P_FLOAT), ctypes.cast(logits_out_address, P_FLOAT)):
        raise ValueError('rwkv_eval_sequence failed, check stderr')