import contextlib
import warnings
import weakref
from typing import ContextManager, List, Optional, Tuple, TYPE_CHECKING
import torch
from torch._C._functorch import (
from torch._guards import Source
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
from torch.utils.weak import WeakIdRef
import torch._prims_common as utils
def meta_tensor(self, t, shape_env=None, callback=lambda t: t(), source: Optional[Source]=None, symbolic_context: Optional['SymbolicContext']=None):
    from torch._subclasses.fake_tensor import FakeTensor
    if source is None:
        from torch._dynamo.source import ConstantSource
        source = ConstantSource(f'__meta_utils_unknown_tensor{len(self.tensor_memo)}')
    assert not torch._C._dispatch_tls_local_exclude_set().has(torch._C.DispatchKey.Python)
    arg_cnt = self.arg_cnt
    self.arg_cnt += 1
    maybe_suppress = contextlib.nullcontext
    if shape_env is not None:
        maybe_suppress = shape_env.suppress_guards

    def sym_sizes_strides_storage_offset(t, src) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        if shape_env is not None:
            if isinstance(t, FakeTensor) and t.fake_mode.shape_env is shape_env:
                return (t.size(), t.stride(), t.storage_offset())
            else:
                return shape_env.create_symbolic_sizes_strides_storage_offset(t, src, symbolic_context=symbolic_context)
        else:
            assert symbolic_context is None
        return (t.size(), t.stride(), t.storage_offset())
    self.check_expired_count += 1
    if self.check_expired_count >= self.check_expired_frequency:
        self.check_for_expired_weak_storages()
        self.check_expired_count = 0
    if self.get_tensor_memo(t) is None:
        with torch.inference_mode(t.is_inference()):
            if t.is_sparse:
                is_leaf = safe_is_leaf(t)
                r = callback(lambda: torch.ops.aten._sparse_coo_tensor_with_dims(t.sparse_dim(), t.dense_dim(), t.shape, dtype=t.dtype, layout=torch.sparse_coo, device='meta'))
                assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                r._coalesced_(t.is_coalesced())
                if t.requires_grad:
                    r.requires_grad = True
                if t.requires_grad and (not is_leaf):
                    with torch.enable_grad():
                        r = r.clone()
                        r._coalesced_(t.is_coalesced())
            elif t.is_mkldnn:
                is_leaf = safe_is_leaf(t)
                sizes, strides, _storage_offset = sym_sizes_strides_storage_offset(t, source)
                r = callback(lambda: torch.empty_strided(sizes, strides, dtype=t.dtype, device='meta'))
                assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                if t.requires_grad:
                    r.requires_grad = True
                if t.requires_grad and (not is_leaf):
                    with torch.enable_grad():
                        r = r.clone()
            elif t._is_view():
                assert t._is_view()
                from torch._dynamo.source import AttrSource
                from torch.fx.experimental.symbolic_shapes import DimDynamic, StatelessSymbolicContext
                if shape_env and (not t.is_nested) and (not t._base.is_nested):
                    base_symbolic_context = StatelessSymbolicContext(dynamic_sizes=[DimDynamic.STATIC] * t._base.dim(), constraint_sizes=[None] * t._base.dim())
                else:
                    base_symbolic_context = None
                base = self.meta_tensor(t._base, shape_env, callback, source=AttrSource(source, '_base'), symbolic_context=base_symbolic_context)

                def is_c_of_r(complex_dtype, real_dtype):
                    return utils.is_complex_dtype(complex_dtype) and utils.corresponding_real_dtype(complex_dtype) == real_dtype
                old_exclude = torch._C._dispatch_tls_is_dispatch_key_excluded(torch._C.DispatchKey.ADInplaceOrView)
                torch._C._dispatch_tls_set_dispatch_key_excluded(torch._C.DispatchKey.ADInplaceOrView, False)
                try:
                    if base.dtype == t.dtype:
                        pass
                    elif is_c_of_r(base.dtype, t.dtype):
                        base = torch.view_as_real(base)
                    elif is_c_of_r(t.dtype, base.dtype):
                        base = torch.view_as_complex(base)
                    else:
                        base = base.view(t.dtype)

                    def _view_from_base(base, t):
                        if t.is_nested:
                            return t._view_func_unsafe(base)
                        else:
                            sizes, strides, storage_offset = sym_sizes_strides_storage_offset(t, source)
                            return base.as_strided(sizes, strides, storage_offset)
                    if safe_is_leaf(t):
                        with torch.no_grad(), maybe_suppress():
                            r = _view_from_base(base, t)
                        r.requires_grad = t.requires_grad
                    elif t._base.requires_grad == t.requires_grad:
                        with torch.enable_grad(), maybe_suppress():
                            r = _view_from_base(base, t)
                    else:
                        assert t.requires_grad
                        with torch.no_grad():
                            mid = base.view(base.shape)
                        mid.requires_grad = t.requires_grad
                        with torch.enable_grad(), maybe_suppress():
                            r = _view_from_base(mid, t)
                    torch._C._autograd._set_creation_meta(r, torch._C._autograd._get_creation_meta(t))
                finally:
                    torch._C._dispatch_tls_set_dispatch_key_excluded(torch._C.DispatchKey.ADInplaceOrView, old_exclude)
            else:
                is_leaf = safe_is_leaf(t)
                if not t.is_nested:
                    sizes, strides, storage_offset = sym_sizes_strides_storage_offset(t, source)

                def empty_create(inner_t, inner_src):
                    inner_sizes, inner_strides, inner_storage_offset = sym_sizes_strides_storage_offset(inner_t, inner_src)
                    return torch.empty_strided(inner_sizes, inner_strides, dtype=inner_t.dtype, device='meta')
                if is_traceable_wrapper_subclass(t):
                    from torch._dynamo.source import AttrSource
                    if t.is_nested:
                        from torch._dynamo.source import TensorProperty, TensorPropertySource
                        attrs, ctx = t.__tensor_flatten__()
                        transformed_tensors_dict = {}
                        orig_shape_env = None
                        for attr in attrs:
                            inner_t = getattr(t, attr)
                            if orig_shape_env is None:
                                orig_shape_env = inner_t.fake_mode.shape_env if isinstance(inner_t, FakeTensor) else None
                            transformed_tensors_dict[attr] = callback(lambda: empty_create(inner_t, AttrSource(source, attr)))
                        assert isinstance(ctx, dict)
                        assert 'ragged_size' in ctx
                        assert isinstance(t._size[1], torch.SymInt)
                        if orig_shape_env is shape_env:
                            ctx['ragged_size'] = t._size[1]
                        else:
                            assert t._size[1].node.singleton_int() is not None
                            ctx['ragged_size'] = shape_env.create_symintnode(shape_env.create_symbol(t._size[1], TensorPropertySource(source, TensorProperty.SIZE, 1)), hint=t._size[1])
                        r = type(t).__tensor_unflatten__(transformed_tensors_dict, ctx)
                    else:
                        r = transform_subclass(t, lambda attr, inner_t: callback(lambda: empty_create(inner_t, AttrSource(source, attr))))
                else:
                    r = callback(lambda: torch.empty_strided(sizes, strides, dtype=t.dtype, device='meta'))
                assert safe_is_leaf(r), "the callback you passed in doesn't detach"
                if t.requires_grad:
                    r.requires_grad = t.requires_grad
                    if not is_leaf:
                        with torch.enable_grad():
                            r = r.clone(memory_format=torch.preserve_format)
                if torch._C._functorch.is_functorch_wrapped_tensor(t):
                    return NotImplemented
                s = t.untyped_storage()
                swr = StorageWeakRef(s)
                if swr not in self.storage_memo and (r.is_nested or (r.stride() == strides and r.storage_offset() == storage_offset)):
                    self.storage_memo[swr] = r.untyped_storage()
                else:
                    r_s = self.meta_storage(s, callback=callback)
                    maybe_fake_mgr: ContextManager[None] = contextlib.nullcontext()
                    from torch._subclasses.fake_tensor import in_kernel_invocation_manager, maybe_get_fake_mode
                    mb_fake_mode = maybe_get_fake_mode(r)
                    if mb_fake_mode is not None:
                        maybe_fake_mgr = in_kernel_invocation_manager(mb_fake_mode)
                    with maybe_fake_mgr, torch.no_grad():
                        r.set_(r_s, storage_offset, sizes, strides)
            if safe_grad(t) is not None:
                from torch._dynamo.source import AttrSource
                r.grad = self.meta_tensor(safe_grad(t), shape_env, callback, source=AttrSource(source, 'grad'), symbolic_context=symbolic_context)
            torch._C._set_conj(r, t.is_conj())
            torch._C._set_neg(r, t.is_neg())
        assert_metadata_eq(assert_eq, t, r, skip_symbolic=True)
        self.set_tensor_memo(t, r)
    return self.get_tensor_memo(t)