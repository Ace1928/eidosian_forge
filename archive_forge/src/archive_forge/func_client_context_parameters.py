from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@CachedProperty
def client_context_parameters(self):
    params = self._service_description.get('clientContextParams', {})
    return [ClientContextParameter(name=param_name, type=param_val['type'], documentation=param_val['documentation']) for param_name, param_val in params.items()]