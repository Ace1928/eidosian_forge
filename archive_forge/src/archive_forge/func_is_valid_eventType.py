import abc
from oslo_serialization import jsonutils
import six
def is_valid_eventType(value):
    return value in VALID_EVENTTYPES