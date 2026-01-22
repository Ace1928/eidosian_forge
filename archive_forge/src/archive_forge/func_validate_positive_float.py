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
def validate_positive_float(option: str, value: Any) -> float:
    """Validates that 'value' is a float, or can be converted to one, and is
    positive.
    """
    errmsg = f'{option} must be an integer or float'
    try:
        value = float(value)
    except ValueError:
        raise ValueError(errmsg) from None
    except TypeError:
        raise TypeError(errmsg) from None
    if not 0 < value < 1000000000.0:
        raise ValueError(f'{option} must be greater than 0 and less than one billion')
    return value