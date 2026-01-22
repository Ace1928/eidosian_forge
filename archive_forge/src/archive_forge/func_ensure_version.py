from __future__ import annotations
import os
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from . import autogenerate as autogen
from . import util
from .runtime.environment import EnvironmentContext
from .script import ScriptDirectory
def ensure_version(config: Config, sql: bool=False) -> None:
    """Create the alembic version table if it doesn't exist already .

    :param config: a :class:`.Config` instance.

    :param sql: use ``--sql`` mode

     .. versionadded:: 1.7.6

    """
    script = ScriptDirectory.from_config(config)

    def do_ensure_version(rev, context):
        context._ensure_version_table()
        return []
    with EnvironmentContext(config, script, fn=do_ensure_version, as_sql=sql):
        script.run_env()