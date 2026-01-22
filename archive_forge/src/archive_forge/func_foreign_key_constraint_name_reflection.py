from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def foreign_key_constraint_name_reflection(self):
    """Target supports reflection of FOREIGN KEY constraints and
        will return the name of the constraint that was used in the
        "CONSTRAINT <name> FOREIGN KEY" DDL.

        MySQL prior to version 8 and MariaDB prior to version 10.5
        don't support this.

        """
    return exclusions.closed()