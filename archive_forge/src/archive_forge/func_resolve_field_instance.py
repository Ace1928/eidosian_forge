from __future__ import annotations
import collections
import datetime as dt
import functools
import inspect
import json
import re
import typing
import warnings
from collections.abc import Mapping
from email.utils import format_datetime, parsedate_to_datetime
from pprint import pprint as py_pprint
from marshmallow.base import FieldABC
from marshmallow.exceptions import FieldInstanceResolutionError
from marshmallow.warnings import RemovedInMarshmallow4Warning
def resolve_field_instance(cls_or_instance):
    """Return a Schema instance from a Schema class or instance.

    :param type|Schema cls_or_instance: Marshmallow Schema class or instance.
    """
    if isinstance(cls_or_instance, type):
        if not issubclass(cls_or_instance, FieldABC):
            raise FieldInstanceResolutionError
        return cls_or_instance()
    else:
        if not isinstance(cls_or_instance, FieldABC):
            raise FieldInstanceResolutionError
        return cls_or_instance