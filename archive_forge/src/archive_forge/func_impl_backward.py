import dataclasses
import functools
import inspect
import sys
import typing
import weakref
from torchgen.model import FunctionSchema, OperatorName, SchemaKind, BaseType, ListType, BaseTy
import torch
import torch._C as _C
import torch.library as library
from torch._library.abstract_impl import AbstractImplCtx
from torch.library import get_ctx
from .autograd import autograd_kernel_indirection, construct_autograd_kernel
def impl_backward(self, output_differentiability=None, _stacklevel=2):
    """Registers a backward formula.

        WARNING: if you're a user, please do not use this directly
        (instead use the torch._custom_ops APIs).
        Also please see the following for a detailed guide on custom ops.
        https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

        In order for the CustomOp to work with autograd, you need to register
        a backward formula. There are two pieces to this:
        1. You must give us a function to specify what to save for backward.
           Call this the "save for backward" function.
        2. You must give us a function that computes gradients. Call this the
           "backward" function.

        Use `impl_save_for_backward` to define a "save for backward" function
        that specifies what gets saved for backward. The function should accept
        two arguments ``(inputs, output)`` and return the quantities to be saved
        for backward.

        During runtime, when you call the CustomOp, PyTorch will invoke the
        "save for backward" function with the inputs and output of the CustomOp.

        Use `impl_backward` to define the "backward" function. The backward
        function must accept ``(ctx, saved, *grads)``:
        - ``ctx`` is a context object where we may provide information
        - ``saved`` is exactly what gets returned from the "save for backward"
          function
        - ``grads`` is one or more gradients. The number of gradients matches
          the number of outputs of the CustomOp.

        The backward function must return a dict that maps the name of
        an input to the CustomOp to its corresponding gradient. All inputs that
        were declared to be Tensors in the CustomOp definition must be accounted
        for in the dict. The gradient may be a Tensor or None.

        """
    if output_differentiability is not None:

        def yell():
            raise RuntimeError(f'impl_backward(output_differentiability): expected output_differentiability to be a list of bools with length equal to the number of outputs of this CustomOp got: {output_differentiability}')
        if not isinstance(output_differentiability, list):
            yell()
        for diff in output_differentiability:
            if not isinstance(diff, bool):
                yell()
        if len(self._schema.returns) != len(output_differentiability):
            yell()

    def inner(f):
        self._check_can_register_backward()
        self._check_doesnt_have_library_autograd_impl()
        if not self._registered_autograd_kernel_indirection:
            self._register_autograd_kernel_indirection()
        self._register_impl('backward', f, stacklevel=_stacklevel)
        self._output_differentiability = output_differentiability
        if self._has_impl('save_for_backward'):
            self._register_autograd_kernel()
    return inner