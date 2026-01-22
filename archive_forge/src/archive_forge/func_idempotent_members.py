from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@CachedProperty
def idempotent_members(self):
    input_shape = self.input_shape
    if not input_shape:
        return []
    return [name for name, shape in input_shape.members.items() if 'idempotencyToken' in shape.metadata and shape.metadata['idempotencyToken']]