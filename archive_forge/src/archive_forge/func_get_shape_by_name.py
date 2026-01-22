from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def get_shape_by_name(self, shape_name, member_traits=None):
    raise ValueError(f"Attempted to lookup shape '{shape_name}', but no shape map was provided.")