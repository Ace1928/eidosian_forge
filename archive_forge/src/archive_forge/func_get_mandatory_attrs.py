from datetime import datetime
from wsme import types as wsme_types
from glance.common import timeutils
@classmethod
def get_mandatory_attrs(cls):
    return [attr.name for attr in cls._wsme_attributes if attr.mandatory]