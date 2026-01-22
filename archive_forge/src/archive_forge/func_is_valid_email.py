import contextlib
import datetime
import ipaddress
import json
import math
from fractions import Fraction
from typing import Callable, Dict, Type, Union, cast, overload
import hypothesis.strategies as st
import pydantic
import pydantic.color
import pydantic.types
from pydantic.utils import lenient_issubclass
def is_valid_email(s: str) -> bool:
    try:
        email_validator.validate_email(s, check_deliverability=False)
        return True
    except email_validator.EmailNotValidError:
        return False