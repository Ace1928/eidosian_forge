import logging
import json
import re
from datetime import date, datetime, time, timezone
import traceback
import importlib
from typing import Any, Dict, Optional, Union, List, Tuple
from inspect import istraceback
from collections import OrderedDict
def merge_record_extra(record: logging.LogRecord, target: Dict, reserved: Union[Dict, List], rename_fields: Optional[Dict[str, str]]=None) -> Dict:
    """
    Merges extra attributes from LogRecord object into target dictionary

    :param record: logging.LogRecord
    :param target: dict to update
    :param reserved: dict or list with reserved keys to skip
    :param rename_fields: an optional dict, used to rename field names in the output.
            Rename levelname to log.level: {'levelname': 'log.level'}
    """
    if rename_fields is None:
        rename_fields = {}
    for key, value in record.__dict__.items():
        if key not in reserved and (not (hasattr(key, 'startswith') and key.startswith('_'))):
            target[rename_fields.get(key, key)] = value
    return target