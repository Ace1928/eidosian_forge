from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
import os
from typing import Any
from googlecloudsdk.api_lib.container import kubeconfig as container_kubeconfig
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
Merge another kubeconfig into self.

    By default, in case of overlapping keys, the value in self is kept and the
    value in the other kubeconfig is lost.

    Args:
      kubeconfig: a Kubeconfig instance
      overwrite: whether to overwrite overlapping keys in self with data from
        the other kubeconfig.
    