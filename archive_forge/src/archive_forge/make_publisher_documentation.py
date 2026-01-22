from typing import Optional, Mapping
from google.api_core.client_options import ClientOptions
from google.auth.credentials import Credentials
from google.cloud.pubsub_v1.types import BatchSettings
from google.cloud.pubsublite.cloudpubsub.internal.async_publisher_impl import (
from google.cloud.pubsublite.cloudpubsub.internal.publisher_impl import (
from google.cloud.pubsublite.cloudpubsub.internal.single_publisher import (
from google.cloud.pubsublite.internal.publisher_client_id import PublisherClientId
from google.cloud.pubsublite.internal.wire.make_publisher import (
from google.cloud.pubsublite.internal.wire.merge_metadata import merge_metadata
from google.cloud.pubsublite.internal.wire.pubsub_context import pubsub_context
from google.cloud.pubsublite.types import TopicPath

    Make a new publisher for the given topic.

    Args:
      topic: The topic to publish to.
      transport: The transport type to use.
      per_partition_batching_settings: Settings for batching messages on each partition. The default is reasonable for most cases.
      credentials: The credentials to use to connect. GOOGLE_DEFAULT_CREDENTIALS is used if None.
      client_options: Other options to pass to the client. Note that if you pass any you must set api_endpoint.
      metadata: Additional metadata to send with the RPC.
      client_id: 128-bit unique client id. If set, enables publish idempotency for the session.

    Returns:
      A new Publisher.

    Throws:
      GoogleApiCallException on any error determining topic structure.
    