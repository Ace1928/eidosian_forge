from typing import Mapping
from urllib.parse import urlencode
from google.cloud.pubsublite.types import Partition, TopicPath, SubscriptionPath
def topic_routing_metadata(topic: TopicPath, partition: Partition) -> Mapping[str, str]:
    encoded = urlencode({'partition': str(partition.value), 'topic': str(topic)})
    return {_PARAMS_HEADER: encoded}