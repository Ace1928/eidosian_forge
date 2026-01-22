from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def shape_for(self, shape_name, member_traits=None):
    return self._shape_resolver.get_shape_by_name(shape_name, member_traits)