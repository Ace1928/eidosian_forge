import yaml
from heat.common.i18n import _
from heat.common import template_format
def default_for_missing(env):
    """Checks a parsed environment for missing sections."""
    for param in SECTIONS:
        if param not in env and param != PARAMETER_MERGE_STRATEGIES:
            if param in (ENCRYPTED_PARAM_NAMES, EVENT_SINKS):
                env[param] = []
            else:
                env[param] = {}