from __future__ import annotations
import re
from kombu.utils.text import escape_regex
def prepare_bind(self, queue, exchange, routing_key, arguments):
    return (routing_key, self.key_to_pattern(routing_key), queue)