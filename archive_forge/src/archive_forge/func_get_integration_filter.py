from __future__ import annotations
import collections.abc as c
import contextlib
import datetime
import json
import os
import re
import shutil
import tempfile
import time
import typing as t
from ...encoding import (
from ...ansible_util import (
from ...executor import (
from ...python_requirements import (
from ...ci import (
from ...target import (
from ...config import (
from ...io import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...cache import (
from .cloud import (
from ...data import (
from ...host_configs import (
from ...host_profiles import (
from ...provisioning import (
from ...pypi_proxy import (
from ...inventory import (
from .filters import (
from .coverage import (
def get_integration_filter(args: IntegrationConfig, targets: list[IntegrationTarget]) -> set[str]:
    """Return a list of test targets to skip based on the host(s) that will be used to run the specified test targets."""
    invalid_targets = sorted((target.name for target in targets if target.target_type not in (IntegrationTargetType.CONTROLLER, IntegrationTargetType.TARGET)))
    if invalid_targets and (not args.list_targets):
        message = f'Unable to determine context for the following test targets: {', '.join(invalid_targets)}\n\nMake sure the test targets are correctly named:\n\n - Modules - The target name should match the module name.\n - Plugins - The target name should be "{{plugin_type}}_{{plugin_name}}".\n\nIf necessary, context can be controlled by adding entries to the "aliases" file for a test target:\n\n - Add the name(s) of modules which are tested.\n - Add "context/target" for module and module_utils tests (these will run on the target host).\n - Add "context/controller" for other test types (these will run on the controller).'
        raise ApplicationError(message)
    invalid_targets = sorted((target.name for target in targets if target.actual_type not in (IntegrationTargetType.CONTROLLER, IntegrationTargetType.TARGET)))
    if invalid_targets:
        if data_context().content.is_ansible:
            display.warning(f'Unable to determine context for the following test targets: {', '.join(invalid_targets)}')
        else:
            display.warning(f'Unable to determine context for the following test targets, they will be run on the target host: {', '.join(invalid_targets)}')
    exclude: set[str] = set()
    controller_targets = [target for target in targets if target.target_type == IntegrationTargetType.CONTROLLER]
    target_targets = [target for target in targets if target.target_type == IntegrationTargetType.TARGET]
    controller_filter = get_target_filter(args, [args.controller], True)
    target_filter = get_target_filter(args, args.targets, False)
    controller_filter.filter_targets(controller_targets, exclude)
    target_filter.filter_targets(target_targets, exclude)
    return exclude