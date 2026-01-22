import os
import multiprocessing
from typing import TypeVar, Optional, Tuple, List
def eval_sequence_in_chunks(self, tokens: List[int], state_in: Optional[NumpyArrayOrPyTorchTensor], state_out: Optional[NumpyArrayOrPyTorchTensor]=None, logits_out: Optional[NumpyArrayOrPyTorchTensor]=None, chunk_size: int=16, use_numpy: bool=False) -> Tuple[NumpyArrayOrPyTorchTensor, NumpyArrayOrPyTorchTensor]:
    """
        Evaluates the model for a sequence of tokens using `eval_sequence`, splitting a potentially long sequence into fixed-length chunks.
        This function is useful for processing complete prompts and user input in chat & role-playing use-cases.
        It is recommended to use this function instead of `eval_sequence` to avoid mistakes and get maximum performance.

        Chunking allows processing sequences of thousands of tokens, while not reaching the ggml's node limit and not consuming too much memory.
        A reasonable and recommended value of chunk size is 16. If you want maximum performance, try different chunk sizes in range [2..64]
        and choose one that works the best in your use case.

        In case of any error, this method will throw an exception.

        Parameters
        ----------
        tokens : List[int]
            Indices of the next tokens to be seen by the model. Must be in range 0 <= token < n_vocab.
        chunk_size : int
            Size of each chunk in tokens, must be positive.
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
    self._library.rwkv_eval_sequence_in_chunks(self._ctx, tokens, chunk_size, state_in_ptr, self._get_data_ptr(state_out), self._get_data_ptr(logits_out))
    return (logits_out, state_out)