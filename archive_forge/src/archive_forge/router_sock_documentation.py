from typing import TYPE_CHECKING, Optional
from ..lib.mailbox import Mailbox
from ..lib.sock_client import SockClient, SockClientClosedError
from .router import MessageRouter, MessageRouterClosedError
Router - handle message router (sock).

Router to manage responses from a socket client.

