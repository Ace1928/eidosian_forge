from typing import List, Optional
from google.cloud.pubsublite_v1 import FlowControlRequest, SequencedMessage
def to_optional(self) -> Optional[FlowControlRequest]:
    allowed_messages = _clamp(self._request.allowed_messages)
    allowed_bytes = _clamp(self._request.allowed_bytes)
    if allowed_messages == 0 and allowed_bytes == 0:
        return None
    request = FlowControlRequest()
    request._pb.allowed_messages = allowed_messages
    request._pb.allowed_bytes = allowed_bytes
    return request