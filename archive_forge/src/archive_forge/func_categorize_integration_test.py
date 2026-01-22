from __future__ import annotations
import collections
import collections.abc as c
import enum
import os
import re
import itertools
import abc
import typing as t
from .encoding import (
from .io import (
from .util import (
from .data import (
def categorize_integration_test(name: str, aliases: list[str], force_target: bool) -> tuple[IntegrationTargetType, IntegrationTargetType]:
    """Return the integration test target types (used and actual) based on the given target name and aliases."""
    context_controller = f'context/{IntegrationTargetType.CONTROLLER.name.lower()}' in aliases
    context_target = f'context/{IntegrationTargetType.TARGET.name.lower()}' in aliases or force_target
    actual_type = None
    strict_mode = data_context().content.is_ansible
    if context_controller and context_target:
        target_type = IntegrationTargetType.CONFLICT
    elif context_controller and (not context_target):
        target_type = IntegrationTargetType.CONTROLLER
    elif context_target and (not context_controller):
        target_type = IntegrationTargetType.TARGET
    else:
        target_types = {IntegrationTargetType.TARGET if plugin_type in ('modules', 'module_utils') else IntegrationTargetType.CONTROLLER for plugin_type, plugin_name in extract_plugin_references(name, aliases)}
        if len(target_types) == 1:
            target_type = target_types.pop()
        elif not target_types:
            actual_type = IntegrationTargetType.UNKNOWN
            target_type = actual_type if strict_mode else IntegrationTargetType.TARGET
        else:
            target_type = IntegrationTargetType.CONFLICT
    return (target_type, actual_type or target_type)