import json
from django.contrib.messages.storage.base import BaseStorage
from django.contrib.messages.storage.cookie import MessageDecoder, MessageEncoder
from django.core.exceptions import ImproperlyConfigured
def deserialize_messages(self, data):
    if data and isinstance(data, str):
        return json.loads(data, cls=MessageDecoder)
    return data