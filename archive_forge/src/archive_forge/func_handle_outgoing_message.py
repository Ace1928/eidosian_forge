from abc import ABC, abstractmethod
from typing import Any, List
@abstractmethod
def handle_outgoing_message(self, stream: str, outgoing_msg: List[Any]) -> None:
    """Broker outgoing ZMQ messages to the kernel websocket."""