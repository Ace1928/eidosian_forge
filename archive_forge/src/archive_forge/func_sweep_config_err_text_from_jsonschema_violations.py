import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import yaml
import wandb
from wandb import util
from wandb.sdk.launch.errors import LaunchError
def sweep_config_err_text_from_jsonschema_violations(violations: List[str]) -> str:
    """Consolidate schema violation strings from wandb/sweeps into a single string.

    Parameters
    ----------
    violations: list of str
        The warnings to render.

    Returns:
    -------
    violation: str
        The consolidated violation text.

    """
    violation_base = 'Malformed sweep config detected! This may cause your sweep to behave in unexpected ways.\nTo avoid this, please fix the sweep config schema violations below:'
    for i, warning in enumerate(violations):
        violations[i] = f'  Violation {i + 1}. {warning}'
    violation = '\n'.join([violation_base] + violations)
    return violation