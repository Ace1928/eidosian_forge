import time
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
def pod_ready(pod: ResourceInstance) -> bool:
    return bool(pod.status and pod.status.containerStatuses is not None and all((container.ready for container in pod.status.containerStatuses)))