from __future__ import annotations
from typing import Union, cast
from uvicorn._types import ASGI3Application, ASGIReceiveCallable, ASGISendCallable, HTTPScope, Scope, WebSocketScope
def get_trusted_client_host(self, x_forwarded_for_hosts: list[str]) -> str | None:
    if self.always_trust:
        return x_forwarded_for_hosts[0]
    for host in reversed(x_forwarded_for_hosts):
        if host not in self.trusted_hosts:
            return host
    return None