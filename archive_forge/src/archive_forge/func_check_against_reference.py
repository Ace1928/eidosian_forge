import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import TestCase, \
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from itertools import chain
from typing import List, Union
from torch._C import TensorType
import io
def check_against_reference(self, func, reference_func, output_func, args, kwargs=None, allow_unused=True, check_types=True, no_grad=False, no_gradgrad=False):
    """Verifies a function performs identically to some reference implementation.

    Commonly, this is used to verify that a JIT implementation
    (output_func) matches the behavior of the eager implementation
    (reference_func).
    """
    kwargs = kwargs if kwargs else {}

    def allSum(vs):
        if isinstance(vs, torch.Tensor):
            vs = (vs,)
        return sum(((i + 1) * v.sum().abs() if v.dtype.is_complex else (i + 1) * v.sum() for i, v in enumerate(vs) if v is not None and v.dtype in floating_and_complex_types_and(torch.half, torch.bfloat16)))

    def clone_tensor(t, preserve_requires_grad):
        require_grad = preserve_requires_grad and t.requires_grad
        return t.detach().clone().requires_grad_(require_grad)

    def clone_inputs(preserve_requires_grad: bool):
        inputs: List[Union[torch.Tensor, List[torch.Tensor]]] = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                inputs.append(clone_tensor(arg, preserve_requires_grad))
            elif is_iterable_of_tensors(arg):
                inputs.append([clone_tensor(t, preserve_requires_grad) for t in arg])
            else:
                inputs.append(arg)
        return inputs

    def get_recording_tensors(args):
        recording_tensors: List[torch.Tensor] = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                recording_tensors.append(arg)
            elif is_iterable_of_tensors(arg):
                recording_tensors.extend(filter(lambda t: t.requires_grad, arg))
        return recording_tensors
    nograd_inputs = clone_inputs(preserve_requires_grad=False)
    outputs = self.runAndSaveRNG(reference_func, nograd_inputs, kwargs)
    with enable_profiling_mode_for_profiling_tests():
        outputs_test = self.runAndSaveRNG(func, nograd_inputs, kwargs)
    self.assertEqual(outputs, outputs_test)
    if check_types:
        check_output_types(self, func, outputs_test, nograd_inputs, kwargs)
    if no_grad:
        return
    with enable_profiling_mode_for_profiling_tests():
        recording_inputs = clone_inputs(preserve_requires_grad=True)
        recording_tensors = get_recording_tensors(recording_inputs)
        outputs = output_func(self.runAndSaveRNG(reference_func, recording_inputs, kwargs))
        grads = torch.autograd.grad(allSum(outputs), recording_tensors, allow_unused=allow_unused)
        outputs_test = output_func(self.runAndSaveRNG(func, recording_inputs, kwargs))
        grads_test = torch.autograd.grad(allSum(outputs_test), recording_tensors, allow_unused=allow_unused)
        self.assertEqual(outputs, outputs_test)
        self.assertEqual(grads, grads_test)
        if self._testMethodName in nn_functional_single_grad or no_gradgrad:
            return
        outputs = output_func(self.runAndSaveRNG(reference_func, recording_inputs, kwargs))
        l1 = allSum(outputs)
        grads = torch.autograd.grad(l1, recording_tensors, create_graph=True, allow_unused=allow_unused)
        l2 = allSum(grads) * l1
        grads2 = torch.autograd.grad(l2, recording_tensors, allow_unused=allow_unused)
        recording_inputs = clone_inputs(preserve_requires_grad=True)
        recording_tensors = get_recording_tensors(recording_inputs)
        outputs_test = output_func(self.runAndSaveRNG(func, recording_inputs, kwargs))
        l1_test = allSum(outputs_test)
        grads_test = torch.autograd.grad(l1_test, recording_tensors, create_graph=True, allow_unused=allow_unused)
        l2_test = allSum(grads_test) * l1_test
        grads2_test = torch.autograd.grad(l2_test, recording_tensors, allow_unused=allow_unused)
        self.assertEqual(outputs, outputs_test)
        self.assertEqual(grads, grads_test)
        for g2, g2_test in zip(grads2, grads2_test):
            if g2 is None and g2_test is None:
                continue
            self.assertEqual(g2, g2_test, atol=0.0005, rtol=0.0001)