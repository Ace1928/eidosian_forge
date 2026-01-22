import os
from json import JSONDecodeError, loads
from typing import Dict, List, Optional, Tuple, Union
from redis.exceptions import DataError
from redis.utils import deprecated_function
from ._util import JsonType
from .decoders import decode_dict_keys
from .path import Path
@deprecated_function(version='4.0.0', reason='deprecated since redisjson 1.0.0')
def nummultby(self, name: str, path: str, number: int) -> str:
    """Multiply the numeric (integer or floating point) JSON value under
        ``path`` at key ``name`` with the provided ``number``.

        For more information see `JSON.NUMMULTBY <https://redis.io/commands/json.nummultby>`_.
        """
    return self.execute_command('JSON.NUMMULTBY', name, str(path), self._encode(number))