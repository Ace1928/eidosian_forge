from __future__ import annotations
import functools
import hmac
import http
from typing import Any, Awaitable, Callable, Iterable, Optional, Tuple, Union, cast
from ..datastructures import Headers
from ..exceptions import InvalidHeader
from ..headers import build_www_authenticate_basic, parse_authorization_basic
from .server import HTTPResponse, WebSocketServerProtocol
def basic_auth_protocol_factory(realm: Optional[str]=None, credentials: Optional[Union[Credentials, Iterable[Credentials]]]=None, check_credentials: Optional[Callable[[str, str], Awaitable[bool]]]=None, create_protocol: Optional[Callable[..., BasicAuthWebSocketServerProtocol]]=None) -> Callable[..., BasicAuthWebSocketServerProtocol]:
    """
    Protocol factory that enforces HTTP Basic Auth.

    :func:`basic_auth_protocol_factory` is designed to integrate with
    :func:`~websockets.server.serve` like this::

        websockets.serve(
            ...,
            create_protocol=websockets.basic_auth_protocol_factory(
                realm="my dev server",
                credentials=("hello", "iloveyou"),
            )
        )

    Args:
        realm: Scope of protection. It should contain only ASCII characters
            because the encoding of non-ASCII characters is undefined.
            Refer to section 2.2 of :rfc:`7235` for details.
        credentials: Hard coded authorized credentials. It can be a
            ``(username, password)`` pair or a list of such pairs.
        check_credentials: Coroutine that verifies credentials.
            It receives ``username`` and ``password`` arguments
            and returns a :class:`bool`. One of ``credentials`` or
            ``check_credentials`` must be provided but not both.
        create_protocol: Factory that creates the protocol. By default, this
            is :class:`BasicAuthWebSocketServerProtocol`. It can be replaced
            by a subclass.
    Raises:
        TypeError: If the ``credentials`` or ``check_credentials`` argument is
            wrong.

    """
    if (credentials is None) == (check_credentials is None):
        raise TypeError('provide either credentials or check_credentials')
    if credentials is not None:
        if is_credentials(credentials):
            credentials_list = [cast(Credentials, credentials)]
        elif isinstance(credentials, Iterable):
            credentials_list = list(credentials)
            if not all((is_credentials(item) for item in credentials_list)):
                raise TypeError(f'invalid credentials argument: {credentials}')
        else:
            raise TypeError(f'invalid credentials argument: {credentials}')
        credentials_dict = dict(credentials_list)

        async def check_credentials(username: str, password: str) -> bool:
            try:
                expected_password = credentials_dict[username]
            except KeyError:
                return False
            return hmac.compare_digest(expected_password, password)
    if create_protocol is None:
        create_protocol = BasicAuthWebSocketServerProtocol
    return functools.partial(create_protocol, realm=realm, check_credentials=check_credentials)