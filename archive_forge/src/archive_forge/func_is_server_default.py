from __future__ import annotations
from typing import Any, Optional, Union
from pymongo.errors import ConfigurationError
@property
def is_server_default(self) -> bool:
    """Does this WriteConcern match the server default."""
    return self.__server_default