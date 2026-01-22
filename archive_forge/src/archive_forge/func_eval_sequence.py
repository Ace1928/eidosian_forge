import os
import multiprocessing
from typing import TypeVar, Optional, Tuple, List
def eval_sequence(self, tokens: List[int], state_in: Optional[NumpyArrayOrPyTorchTensor], state_out: Optional[NumpyArrayOrPyTorchTensor]=None, logits_out: Optional[NumpyArrayOrPyTorchTensor]=None, use_numpy: bool=False) -> Tuple[NumpyArrayOrPyTorchTensor, NumpyArrayOrPyTorchTensor]:
    """
        Evaluates the model for a sequence of tokens.

        NOTE ON GGML NODE LIMIT

        ggml has a hard-coded limit on max amount of nodes in a computation graph. The sequence graph is built in a way that quickly exceedes
        this limit when using large models and/or large sequence lengths.
        Fortunately, rwkv.cpp's fork of ggml has increased limit which was tested to work for sequence lengths up to 64 for 14B models.

        If you get `GGML_ASSERT: ...\\ggml.c:16941: cgraph->n_nodes < GGML_MAX_NODES`, this means you've exceeded the limit.
        To get rid of the assertion failure, reduce the model size and/or sequence length.

        In case of any error, this method will throw an exception.

        Parameters
        ----------
        tokens : List[int]
            Indices of the next tokens to be seen by the model. Must be in range 0 <= token < n_vocab.
        state_in : Optional[NumpyArrayOrTorchTensor]
            State from previous call of this method. If this is a first pass, set it to None.
        state_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for state. If provided, must be of type float32, contiguous and of shape (state_buffer_element_count).
        logits_out : Optional[NumpyArrayOrTorchTensor]
            Optional output tensor for logits. If provided, must be of type float32, contiguous and of shape (logits_buffer_element_count).
        use_numpy : bool
            If set to True, numpy's ndarrays will be created instead of PyTorch's Tensors.
            This parameter is ignored if any tensor parameter is not None; in such case,
            type of returned tensors will match the type of received tensors.

        Returns
        -------
        logits, state
            Logits vector of shape (n_vocab); state for the next step.
        """
    if not self._valid:
        raise ValueError('Model was freed')
    use_numpy = self._detect_numpy_usage([state_in, state_out, logits_out], use_numpy)
    if state_in is not None:
        self._validate_tensor(state_in, 'state_in', self._state_buffer_element_count)
        state_in_ptr = self._get_data_ptr(state_in)
    else:
        state_in_ptr = 0
    if state_out is not None:
        self._validate_tensor(state_out, 'state_out', self._state_buffer_element_count)
    else:
        state_out = self._zeros_float32(self._state_buffer_element_count, use_numpy)
    if logits_out is not None:
        self._validate_tensor(logits_out, 'logits_out', self._logits_buffer_element_count)
    else:
        logits_out = self._zeros_float32(self._logits_buffer_element_count, use_numpy)
    self._library.rwkv_eval_sequence(self._ctx, tokens, state_in_ptr, self._get_data_ptr(state_out), self._get_data_ptr(logits_out))
    return (logits_out, state_out)