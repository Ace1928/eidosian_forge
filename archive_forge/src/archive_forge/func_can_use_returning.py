from __future__ import annotations
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import context
from . import evaluator
from . import exc as orm_exc
from . import loading
from . import persistence
from .base import NO_VALUE
from .context import AbstractORMCompileState
from .context import FromStatement
from .context import ORMFromStatementCompileState
from .context import QueryContext
from .. import exc as sa_exc
from .. import util
from ..engine import Dialect
from ..engine import result as _result
from ..sql import coercions
from ..sql import dml
from ..sql import expression
from ..sql import roles
from ..sql import select
from ..sql import sqltypes
from ..sql.base import _entity_namespace_key
from ..sql.base import CompileState
from ..sql.base import Options
from ..sql.dml import DeleteDMLState
from ..sql.dml import InsertDMLState
from ..sql.dml import UpdateDMLState
from ..util import EMPTY_DICT
from ..util.typing import Literal
@classmethod
def can_use_returning(cls, dialect: Dialect, mapper: Mapper[Any], *, is_multitable: bool=False, is_update_from: bool=False, is_delete_using: bool=False, is_executemany: bool=False) -> bool:
    normal_answer = dialect.delete_returning and mapper.local_table.implicit_returning
    if not normal_answer:
        return False
    if is_delete_using:
        return dialect.delete_returning_multifrom
    elif is_multitable and (not dialect.delete_returning_multifrom):
        raise sa_exc.CompileError(f'''Dialect "{dialect.name}" does not support RETURNING with DELETE..USING; for synchronize_session='fetch', please add the additional execution option 'is_delete_using=True' to the statement to indicate that a separate SELECT should be used for this backend.''')
    return True