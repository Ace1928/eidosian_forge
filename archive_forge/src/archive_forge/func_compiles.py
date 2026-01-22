from .. import exc
from ..sql import sqltypes
def compiles(class_, *specs):
    """Register a function as a compiler for a
    given :class:`_expression.ClauseElement` type."""

    def decorate(fn):
        existing = class_.__dict__.get('_compiler_dispatcher', None)
        existing_dispatch = getattr(class_, '_compiler_dispatch', None)
        if not existing:
            existing = _dispatcher()
            if existing_dispatch:

                def _wrap_existing_dispatch(element, compiler, **kw):
                    try:
                        return existing_dispatch(element, compiler, **kw)
                    except exc.UnsupportedCompilationError as uce:
                        raise exc.UnsupportedCompilationError(compiler, type(element), message='%s construct has no default compilation handler.' % type(element)) from uce
                existing.specs['default'] = _wrap_existing_dispatch
            setattr(class_, '_compiler_dispatch', lambda *arg, **kw: existing(*arg, **kw))
            setattr(class_, '_compiler_dispatcher', existing)
        if specs:
            for s in specs:
                existing.specs[s] = fn
        else:
            existing.specs['default'] = fn
        return fn
    return decorate