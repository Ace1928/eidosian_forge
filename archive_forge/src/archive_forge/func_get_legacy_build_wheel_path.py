import logging
import os.path
from typing import List, Optional
from pip._internal.cli.spinners import open_spinner
from pip._internal.utils.setuptools_build import make_setuptools_bdist_wheel_args
from pip._internal.utils.subprocess import call_subprocess, format_command_args
def get_legacy_build_wheel_path(names: List[str], temp_dir: str, name: str, command_args: List[str], command_output: str) -> Optional[str]:
    """Return the path to the wheel in the temporary build directory."""
    names = sorted(names)
    if not names:
        msg = 'Legacy build of wheel for {!r} created no files.\n'.format(name)
        msg += format_command_result(command_args, command_output)
        logger.warning(msg)
        return None
    if len(names) > 1:
        msg = 'Legacy build of wheel for {!r} created more than one file.\nFilenames (choosing first): {}\n'.format(name, names)
        msg += format_command_result(command_args, command_output)
        logger.warning(msg)
    return os.path.join(temp_dir, names[0])