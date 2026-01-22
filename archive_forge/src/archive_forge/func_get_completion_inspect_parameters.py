import os
import sys
from typing import Any, MutableMapping, Tuple
import click
from ._completion_classes import completion_init
from ._completion_shared import Shells, get_completion_script, install
from .models import ParamMeta
from .params import Option
from .utils import get_params_from_function
def get_completion_inspect_parameters() -> Tuple[ParamMeta, ParamMeta]:
    completion_init()
    test_disable_detection = os.getenv('_TYPER_COMPLETE_TEST_DISABLE_SHELL_DETECTION')
    if shellingham and (not test_disable_detection):
        parameters = get_params_from_function(_install_completion_placeholder_function)
    else:
        parameters = get_params_from_function(_install_completion_no_auto_placeholder_function)
    install_param, show_param = parameters.values()
    return (install_param, show_param)