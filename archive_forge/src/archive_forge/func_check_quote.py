from __future__ import annotations
import re
from types import ModuleType
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from urllib.parse import unquote_plus
from . import Connector
from .. import ExecutionContext
from .. import pool
from .. import util
from ..engine import ConnectArgsType
from ..engine import Connection
from ..engine import interfaces
from ..engine import URL
from ..sql.type_api import TypeEngine
def check_quote(token: str) -> str:
    if ';' in str(token) or str(token).startswith('{'):
        token = '{%s}' % token.replace('}', '}}')
    return token