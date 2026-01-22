import importlib
from django.apps import apps
from django.conf import settings
from django.core.serializers.base import SerializerDoesNotExist
def get_serializer_formats():
    if not _serializers:
        _load_serializers()
    return list(_serializers)