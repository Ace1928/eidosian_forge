import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import sys
import typing
from typing import (
from . import requirements, specifiers, utils, version as version_module
@classmethod
def from_email(cls, data: Union[bytes, str], *, validate: bool=True) -> 'Metadata':
    """Parse metadata from email headers.

        If *validate* is true, the metadata will be validated. All exceptions
        related to validation will be gathered and raised as an :class:`ExceptionGroup`.
        """
    raw, unparsed = parse_email(data)
    if validate:
        exceptions: list[Exception] = []
        for unparsed_key in unparsed:
            if unparsed_key in _EMAIL_TO_RAW_MAPPING:
                message = f'{unparsed_key!r} has invalid data'
            else:
                message = f'unrecognized field: {unparsed_key!r}'
            exceptions.append(InvalidMetadata(unparsed_key, message))
        if exceptions:
            raise ExceptionGroup('unparsed', exceptions)
    try:
        return cls.from_raw(raw, validate=validate)
    except ExceptionGroup as exc_group:
        raise ExceptionGroup('invalid or unparsed metadata', exc_group.exceptions) from None