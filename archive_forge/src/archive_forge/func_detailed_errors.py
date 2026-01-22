import io
import json
import logging
import os
import re
from contextlib import contextmanager
from textwrap import indent, wrap
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union, cast
from .fastjsonschema_exceptions import JsonSchemaValueException
@contextmanager
def detailed_errors():
    try:
        yield
    except JsonSchemaValueException as ex:
        raise ValidationError._from_jsonschema(ex) from None