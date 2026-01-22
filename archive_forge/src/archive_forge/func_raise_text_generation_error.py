import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
def raise_text_generation_error(http_error: HTTPError) -> NoReturn:
    """
    Try to parse text-generation-inference error message and raise HTTPError in any case.

    Args:
        error (`HTTPError`):
            The HTTPError that have been raised.
    """
    try:
        payload = getattr(http_error, 'response_error_payload', None) or http_error.response.json()
        error = payload.get('error')
        error_type = payload.get('error_type')
    except Exception:
        raise http_error
    if error_type is not None:
        exception = _parse_text_generation_error(error, error_type)
        raise exception from http_error
    raise http_error