import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
def stricturl(*, strip_whitespace: bool=True, min_length: int=1, max_length: int=2 ** 16, tld_required: bool=True, host_required: bool=True, allowed_schemes: Optional[Collection[str]]=None) -> Type[AnyUrl]:
    namespace = dict(strip_whitespace=strip_whitespace, min_length=min_length, max_length=max_length, tld_required=tld_required, host_required=host_required, allowed_schemes=allowed_schemes)
    return type('UrlValue', (AnyUrl,), namespace)