from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def with_lifting(func_ir, typingctx, targetctx, flags, locals):
    """With-lifting transformation

    Rewrite the IR to extract all withs.
    Only the top-level withs are extracted.
    Returns the (the_new_ir, the_lifted_with_ir)
    """
    from numba.core import postproc

    def dispatcher_factory(func_ir, objectmode=False, **kwargs):
        from numba.core.dispatcher import LiftedWith, ObjModeLiftedWith
        myflags = flags.copy()
        if objectmode:
            myflags.enable_looplift = False
            myflags.enable_pyobject = True
            myflags.force_pyobject = True
            myflags.no_cpython_wrapper = False
            cls = ObjModeLiftedWith
        else:
            cls = LiftedWith
        return cls(func_ir, typingctx, targetctx, myflags, locals, **kwargs)
    withs, func_ir = find_setupwiths(func_ir)
    if not withs:
        return (func_ir, [])
    postproc.PostProcessor(func_ir).run()
    assert func_ir.variable_lifetime
    vlt = func_ir.variable_lifetime
    blocks = func_ir.blocks.copy()
    cfg = vlt.cfg
    sub_irs = []
    for blk_start, blk_end in withs:
        body_blocks = []
        for node in _cfg_nodes_in_region(cfg, blk_start, blk_end):
            body_blocks.append(node)
        _legalize_with_head(blocks[blk_start])
        cmkind, extra = _get_with_contextmanager(func_ir, blocks, blk_start)
        sub = cmkind.mutate_with_body(func_ir, blocks, blk_start, blk_end, body_blocks, dispatcher_factory, extra)
        sub_irs.append(sub)
    if not sub_irs:
        new_ir = func_ir
    else:
        new_ir = func_ir.derive(blocks)
    return (new_ir, sub_irs)