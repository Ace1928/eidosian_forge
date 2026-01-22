from __future__ import absolute_import, division, print_function
import copy
from collections import defaultdict
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.service import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.resource import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.runner import (
Module execution