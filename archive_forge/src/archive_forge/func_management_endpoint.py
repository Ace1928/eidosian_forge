import json
import time
import asyncio
import typing
import threading
from contextlib import asynccontextmanager, suppress
from lazyops.utils.logs import default_logger as logger
from aiokeydb.v1.client.types import KeyDBUri, lazyproperty
from lazyops.utils.serialization import ObjectEncoder
from aiokeydb.v1.client.serializers import SerializerType
from aiokeydb.v1.client.meta import KeyDBClient
from aiokeydb.v1.client.schemas.session import KeyDBSession
from aiokeydb.v1.connection import ConnectionPool, BlockingConnectionPool
from aiokeydb.v1.commands.core import AsyncScript
from aiokeydb.v1.asyncio.connection import AsyncConnectionPool, AsyncBlockingConnectionPool
from aiokeydb.v1.queues.errors import JobError
from aiokeydb.v1.queues.types import (
from aiokeydb.v1.queues.utils import (
from lazyops.imports._aiohttpx import (
from aiokeydb.v1.utils import set_ulimits, get_ulimits
@lazyproperty
def management_endpoint(self) -> str:
    """
        Returns the management endpoint
        """
    if self.management_url is None:
        return None
    from urllib.parse import urljoin
    return urljoin(self.management_url, self.management_register_api_path)