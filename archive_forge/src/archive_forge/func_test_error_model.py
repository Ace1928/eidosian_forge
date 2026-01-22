import unittest
from contextlib import contextmanager
from llvmlite import ir
from numba.core import types, typing, callconv, cpu, cgutils
from numba.core.registry import cpu_target
def test_error_model(self):
    """
        Caching must not mix up different error models.
        """

    def inv(x):
        return 1.0 / x
    inv_sig = typing.signature(types.float64, types.float64)

    def compile_inv(context):
        return context.compile_subroutine(builder, inv, inv_sig)
    with self._context_builder_sig_args() as (context, builder, sig, args):
        py_error_model = callconv.create_error_model('python', context)
        np_error_model = callconv.create_error_model('numpy', context)
        py_context1 = context.subtarget(error_model=py_error_model)
        py_context2 = context.subtarget(error_model=py_error_model)
        np_context = context.subtarget(error_model=np_error_model)
        initial_cache_size = len(context.cached_internal_func)
        self.assertEqual(initial_cache_size + 0, len(context.cached_internal_func))
        compile_inv(py_context1)
        self.assertEqual(initial_cache_size + 1, len(context.cached_internal_func))
        compile_inv(py_context2)
        self.assertEqual(initial_cache_size + 1, len(context.cached_internal_func))
        compile_inv(np_context)
        self.assertEqual(initial_cache_size + 2, len(context.cached_internal_func))