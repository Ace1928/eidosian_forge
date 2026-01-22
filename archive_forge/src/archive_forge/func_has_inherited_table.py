from .extensions import AbstractConcreteBase
from .extensions import ConcreteBase
from .extensions import DeferredReflection
from ... import util
from ...orm.decl_api import as_declarative as _as_declarative
from ...orm.decl_api import declarative_base as _declarative_base
from ...orm.decl_api import DeclarativeMeta
from ...orm.decl_api import declared_attr
from ...orm.decl_api import has_inherited_table as _has_inherited_table
from ...orm.decl_api import synonym_for as _synonym_for
@util.moved_20('The ``has_inherited_table()`` function is now available as :func:`sqlalchemy.orm.has_inherited_table`.')
def has_inherited_table(*arg, **kw):
    return _has_inherited_table(*arg, **kw)