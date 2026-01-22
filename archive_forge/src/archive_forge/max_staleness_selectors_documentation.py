from __future__ import annotations
from typing import TYPE_CHECKING
from pymongo.errors import ConfigurationError
from pymongo.server_type import SERVER_TYPE
Apply max_staleness, in seconds, to a Selection.