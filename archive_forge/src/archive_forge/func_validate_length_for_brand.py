import abc
import math
import re
import warnings
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from types import new_class
from typing import (
from uuid import UUID
from weakref import WeakSet
from . import errors
from .datetime_parse import parse_date
from .utils import import_string, update_not_none
from .validators import (
@classmethod
def validate_length_for_brand(cls, card_number: 'PaymentCardNumber') -> 'PaymentCardNumber':
    """
        Validate length based on BIN for major brands:
        https://en.wikipedia.org/wiki/Payment_card_number#Issuer_identification_number_(IIN)
        """
    required_length: Union[None, int, str] = None
    if card_number.brand in PaymentCardBrand.mastercard:
        required_length = 16
        valid = len(card_number) == required_length
    elif card_number.brand == PaymentCardBrand.visa:
        required_length = '13, 16 or 19'
        valid = len(card_number) in {13, 16, 19}
    elif card_number.brand == PaymentCardBrand.amex:
        required_length = 15
        valid = len(card_number) == required_length
    else:
        valid = True
    if not valid:
        raise errors.InvalidLengthForBrand(brand=card_number.brand, required_length=required_length)
    return card_number