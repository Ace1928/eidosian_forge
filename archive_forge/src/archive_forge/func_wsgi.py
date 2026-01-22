import io
import math
import sys
import typing
import warnings
import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from starlette.types import Receive, Scope, Send
def wsgi(self, environ: typing.Dict[str, typing.Any], start_response: typing.Callable[..., typing.Any]) -> None:
    for chunk in self.app(environ, start_response):
        anyio.from_thread.run(self.stream_send.send, {'type': 'http.response.body', 'body': chunk, 'more_body': True})
    anyio.from_thread.run(self.stream_send.send, {'type': 'http.response.body', 'body': b''})