from __future__ import annotations
import contextlib
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import inspect
from . import compare
from . import render
from .. import util
from ..operations import ops
from ..util import sqla_compat
def render_python_code(up_or_down_op: Union[UpgradeOps, DowngradeOps], sqlalchemy_module_prefix: str='sa.', alembic_module_prefix: str='op.', render_as_batch: bool=False, imports: Sequence[str]=(), render_item: Optional[RenderItemFn]=None, migration_context: Optional[MigrationContext]=None, user_module_prefix: Optional[str]=None) -> str:
    """Render Python code given an :class:`.UpgradeOps` or
    :class:`.DowngradeOps` object.

    This is a convenience function that can be used to test the
    autogenerate output of a user-defined :class:`.MigrationScript` structure.

    :param up_or_down_op: :class:`.UpgradeOps` or :class:`.DowngradeOps` object
    :param sqlalchemy_module_prefix: module prefix for SQLAlchemy objects
    :param alembic_module_prefix: module prefix for Alembic constructs
    :param render_as_batch: use "batch operations" style for rendering
    :param imports: sequence of import symbols to add
    :param render_item: callable to render items
    :param migration_context: optional :class:`.MigrationContext`
    :param user_module_prefix: optional string prefix for user-defined types

     .. versionadded:: 1.11.0

    """
    opts = {'sqlalchemy_module_prefix': sqlalchemy_module_prefix, 'alembic_module_prefix': alembic_module_prefix, 'render_item': render_item, 'render_as_batch': render_as_batch, 'user_module_prefix': user_module_prefix}
    if migration_context is None:
        from ..runtime.migration import MigrationContext
        from sqlalchemy.engine.default import DefaultDialect
        migration_context = MigrationContext.configure(dialect=DefaultDialect())
    autogen_context = AutogenContext(migration_context, opts=opts)
    autogen_context.imports = set(imports)
    return render._indent(render._render_cmd_body(up_or_down_op, autogen_context))