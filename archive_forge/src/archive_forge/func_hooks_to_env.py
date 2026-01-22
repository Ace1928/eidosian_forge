from collections import abc
from oslo_serialization import jsonutils
from urllib import error
from urllib import parse
from urllib import request
from heatclient._i18n import _
from heatclient.common import environment_format
from heatclient.common import template_format
from heatclient.common import utils
from heatclient import exc
def hooks_to_env(env, arg_hooks, hook):
    """Add hooks from args to environment's resource_registry section.

    Hooks are either "resource_name" (if it's a top-level resource) or
    "nested_stack/resource_name" (if the resource is in a nested stack).

    The environment expects each hook to be associated with the resource
    within `resource_registry/resources` using the `hooks: pre-create` format.
    """
    if 'resource_registry' not in env:
        env['resource_registry'] = {}
    if 'resources' not in env['resource_registry']:
        env['resource_registry']['resources'] = {}
    for hook_declaration in arg_hooks:
        hook_path = [r for r in hook_declaration.split('/') if r]
        resources = env['resource_registry']['resources']
        for nested_stack in hook_path:
            if nested_stack not in resources:
                resources[nested_stack] = {}
            resources = resources[nested_stack]
        else:
            resources['hooks'] = hook