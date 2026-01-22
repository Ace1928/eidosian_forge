import torch
from collections import OrderedDict
import weakref
import warnings
from typing import Any, Tuple
def setup_output_hook(self, args):

    def fn(grad_fn):

        def hook(_, grad_output):
            self.grad_outputs = self._pack_with_none(self.output_tensors_index, grad_output, self.n_outputs)
            if self.user_pre_hooks:
                expected_len = len(self.grad_outputs)
                for user_pre_hook in self.user_pre_hooks:
                    hook_grad_outputs = user_pre_hook(self.module, self.grad_outputs)
                    if hook_grad_outputs is None:
                        continue
                    actual_len = len(hook_grad_outputs)
                    if actual_len != expected_len:
                        raise RuntimeError(f'Backward pre hook returned an invalid number of grad_output, got {actual_len}, but expected {expected_len}')
                    self.grad_outputs = hook_grad_outputs
            if self.input_tensors_index is None:
                grad_inputs = self._pack_with_none([], [], self.n_inputs)
                for user_hook in self.user_hooks:
                    res = user_hook(self.module, grad_inputs, self.grad_outputs)
                    if res is not None and (not (isinstance(res, tuple) and all((el is None for el in res)))):
                        raise RuntimeError('Backward hook for Modules where no input requires gradient should always return None or None for all gradients.')
                self.grad_outputs = None
            if self.grad_outputs is not None:
                assert self.output_tensors_index is not None
                return tuple((self.grad_outputs[i] for i in self.output_tensors_index))
        grad_fn.register_hook(hook)
    is_tuple = True
    if not isinstance(args, tuple):
        args = (args,)
        is_tuple = False
    res, output_idx = self._apply_on_tensors(fn, args)
    self.n_outputs = len(args)
    self.output_tensors_index = output_idx
    if not is_tuple:
        res = res[0]
    return res