import base64
import pickle
from django.db import models
from django.utils import encoding
import jsonpickle
import oauth2client
def to_python(self, value):
    """Overrides ``models.Field`` method. This is used to convert
        bytes (from serialization etc) to an instance of this class"""
    if value is None:
        return None
    elif isinstance(value, oauth2client.client.Credentials):
        return value
    else:
        try:
            return jsonpickle.decode(base64.b64decode(encoding.smart_bytes(value)).decode())
        except ValueError:
            return pickle.loads(base64.b64decode(encoding.smart_bytes(value)))