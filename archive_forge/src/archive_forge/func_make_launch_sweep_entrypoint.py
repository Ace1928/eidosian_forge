import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import yaml
import wandb
from wandb import util
from wandb.sdk.launch.errors import LaunchError
def make_launch_sweep_entrypoint(args: Dict[str, Any], command: Optional[List[str]]) -> Tuple[Optional[List[str]], Any]:
    """Use args dict from create_sweep_command_args to construct entrypoint.

    If replace is True, remove macros from entrypoint, fill them in with args
    and then return the args in seperate return value.
    """
    if not command:
        return (None, None)
    entry_point = create_sweep_command(command)
    macro_args = {}
    for macro in args:
        mstr = '${' + macro + '}'
        if mstr in entry_point:
            idx = entry_point.index(mstr)
            macro_args = args[macro]
            entry_point = entry_point[:idx] + entry_point[idx + 1:]
    if len(entry_point) == 0:
        return (None, macro_args)
    return (entry_point, macro_args)