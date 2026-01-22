import datetime
from google.api_core.exceptions import InvalidArgument
from cloudsdk.google.protobuf.timestamp_pb2 import Timestamp  # pytype: disable=pyi-error
from google.pubsub_v1 import PubsubMessage
from google.cloud.pubsublite.cloudpubsub import MessageTransformer
from google.cloud.pubsublite.internal import fast_serialize
from google.cloud.pubsublite.types import Partition, MessageMetadata
from google.cloud.pubsublite_v1 import AttributeValues, SequencedMessage, PubSubMessage
def from_cps_publish_message(source: PubsubMessage) -> PubSubMessage:
    source_pb = source._pb
    out = PubSubMessage()
    out_pb = out._pb
    if PUBSUB_LITE_EVENT_TIME in source_pb.attributes:
        out_pb.event_time.CopyFrom(_decode_attribute_event_time_proto(source_pb.attributes[PUBSUB_LITE_EVENT_TIME]))
    out_pb.data = source_pb.data
    out_pb.key = source_pb.ordering_key.encode('utf-8')
    for key, value in source_pb.attributes.items():
        if key != PUBSUB_LITE_EVENT_TIME:
            out_pb.attributes[key].values.append(value.encode('utf-8'))
    return out