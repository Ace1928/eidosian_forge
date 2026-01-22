from .. import core
@core.extern
def globaltimer(_builder=None):
    return core.inline_asm_elementwise('mov.u64 $0, %globaltimer;', '=l', [], dtype=core.int64, is_pure=False, pack=1, _builder=_builder)