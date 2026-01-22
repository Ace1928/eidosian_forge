import itertools
import unittest.mock
from contextlib import contextmanager
from typing import Iterator
import torch
import torch._C
import torch._ops
import torch.utils._python_dispatch
import torch.utils._pytree as pytree
def make_crossref_functionalize(op, final_key):
    from torch._subclasses.fake_tensor import FakeTensorMode
    if op == torch.ops.aten.lift_fresh.default:
        return final_key

    def handler(*args, **kwargs):
        fake_mode = FakeTensorMode()

        def fakeify_defun(t):
            if isinstance(t, torch.Tensor):
                if torch._is_functional_tensor(t):
                    r = torch._from_functional_tensor(t)
                    assert t.size() == r.size()
                    assert t.stride() == r.stride()
                else:
                    r = t
                return fake_mode.from_tensor(r)
            return t

        def maybe_detach(t):
            if isinstance(t, torch.Tensor):
                return t.detach()
            else:
                return t
        with torch.utils._python_dispatch._disable_current_modes(), suspend_functionalization():
            f_args, f_kwargs = pytree.tree_map(fakeify_defun, (args, kwargs))
            orig_f_args, orig_f_kwargs = pytree.tree_map(maybe_detach, (f_args, f_kwargs))
            with fake_mode:
                f_r = op(*f_args, **f_kwargs)
        r = op._op_dk(final_key, *args, **kwargs)

        def desc():
            fmt_args = ', '.join(itertools.chain((repr(pytree.tree_map(_fmt, a)) for a in orig_f_args), (f'{k}={pytree.tree_map(_fmt, v)}' for k, v in orig_f_kwargs.items())))
            return f'{op}({fmt_args})'
        check_metadata_matches(f_r, r, desc)
        return r
    return handler