from concurrent.futures import Future
from typing import Optional, Mapping, Union
from uuid import uuid4
from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
from google.cloud.pubsub_v1.types import BatchSettings
from google.cloud.pubsublite.cloudpubsub.internal.make_publisher import (
from google.cloud.pubsublite.cloudpubsub.internal.multiplexed_async_publisher_client import (
from google.cloud.pubsublite.cloudpubsub.internal.multiplexed_publisher_client import (
from google.cloud.pubsublite.cloudpubsub.publisher_client_interface import (
from google.cloud.pubsublite.internal.constructable_from_service_account import (
from google.cloud.pubsublite.internal.publisher_client_id import PublisherClientId
from google.cloud.pubsublite.internal.require_started import RequireStarted
from google.cloud.pubsublite.internal.wire.make_publisher import (
from google.cloud.pubsublite.types import TopicPath

        Create a new AsyncPublisherClient.

        Args:
            per_partition_batching_settings: The settings for publish batching. Apply on a per-partition basis.
            credentials: If provided, the credentials to use when connecting.
            transport: The transport to use. Must correspond to an asyncio transport.
            client_options: The client options to use when connecting. If used, must explicitly set `api_endpoint`.
            enable_idempotence: Whether idempotence is enabled, where the server will ensure that unique messages within a single publisher session are stored only once.
        