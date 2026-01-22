import collections
import decimal
from io import StringIO
import yaml
from django.core.serializers.base import DeserializationError
from django.core.serializers.python import Deserializer as PythonDeserializer
from django.core.serializers.python import Serializer as PythonSerializer
from django.db import models
Deserialize a stream or string of YAML data.