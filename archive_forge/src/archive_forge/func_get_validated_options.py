from __future__ import annotations
import datetime
import inspect
import warnings
from collections import OrderedDict, abc
from typing import (
from urllib.parse import unquote_plus
from bson import SON
from bson.binary import UuidRepresentation
from bson.codec_options import CodecOptions, DatetimeConversion, TypeRegistry
from bson.raw_bson import RawBSONDocument
from pymongo.auth import MECHANISMS
from pymongo.compression_support import (
from pymongo.driver_info import DriverInfo
from pymongo.errors import ConfigurationError
from pymongo.monitoring import _validate_event_listeners
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import _MONGOS_MODES, _ServerMode
from pymongo.server_api import ServerApi
from pymongo.write_concern import DEFAULT_WRITE_CONCERN, WriteConcern, validate_boolean
def get_validated_options(options: Mapping[str, Any], warn: bool=True) -> MutableMapping[str, Any]:
    """Validate each entry in options and raise a warning if it is not valid.
    Returns a copy of options with invalid entries removed.

    :Parameters:
        - `opts`: A dict containing MongoDB URI options.
        - `warn` (optional): If ``True`` then warnings will be logged and
          invalid options will be ignored. Otherwise, invalid options will
          cause errors.
    """
    validated_options: MutableMapping[str, Any]
    if isinstance(options, _CaseInsensitiveDictionary):
        validated_options = _CaseInsensitiveDictionary()

        def get_normed_key(x: str) -> str:
            return x

        def get_setter_key(x: str) -> str:
            return options.cased_key(x)
    else:
        validated_options = {}

        def get_normed_key(x: str) -> str:
            return x.lower()

        def get_setter_key(x: str) -> str:
            return x
    for opt, value in options.items():
        normed_key = get_normed_key(opt)
        try:
            validator = URI_OPTIONS_VALIDATOR_MAP.get(normed_key, raise_config_error)
            value = validator(opt, value)
        except (ValueError, TypeError, ConfigurationError) as exc:
            if warn:
                warnings.warn(str(exc), stacklevel=2)
            else:
                raise
        else:
            validated_options[get_setter_key(normed_key)] = value
    return validated_options