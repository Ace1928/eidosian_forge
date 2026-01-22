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
def validate_document_class(option: str, value: Any) -> Union[Type[MutableMapping], Type[RawBSONDocument]]:
    """Validate the document_class option."""
    is_mapping = False
    try:
        is_mapping = issubclass(value, abc.MutableMapping)
    except TypeError:
        if hasattr(value, '__origin__'):
            is_mapping = issubclass(value.__origin__, abc.MutableMapping)
    if not is_mapping and (not issubclass(value, RawBSONDocument)):
        raise TypeError(f'{option} must be dict, bson.son.SON, bson.raw_bson.RawBSONDocument, or a subclass of collections.MutableMapping')
    return value