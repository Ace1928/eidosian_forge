import abc
import dataclasses
import enum
import typing
import warnings
from spnego._credential import Credential
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import FeatureMissingError, NegotiateOptions, SpnegoError
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
def split_username(username: typing.Optional[str]) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]:
    """Splits a username and returns the domain component.

    Will split a username in the Netlogon form `DOMAIN\\username` and return the domain and user part as separate
    strings. If the user does not contain the `DOMAIN\\` prefix or is in the `UPN` form then then user stays the same
    and the domain is an empty string.

    Args:
        username: The username to split

    Returns:
        Tuple[Optional[str], Optional[str]]: The domain and username.
    """
    if username is None:
        return (None, None)
    domain: typing.Optional[str]
    if '\\' in username:
        domain, username = username.split('\\', 1)
    else:
        domain = None
    return (to_text(domain, nonstring='passthru'), to_text(username, nonstring='passthru'))