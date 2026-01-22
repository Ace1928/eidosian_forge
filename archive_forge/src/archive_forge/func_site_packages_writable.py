import errno
import json
import operator
import os
import shutil
import site
from optparse import SUPPRESS_HELP, Values
from typing import List, Optional
from pip._vendor.rich import print_json
from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.req_command import (
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, InstallationError
from pip._internal.locations import get_scheme
from pip._internal.metadata import get_environment
from pip._internal.models.installation_report import InstallationReport
from pip._internal.operations.build.build_tracker import get_build_tracker
from pip._internal.operations.check import ConflictDetails, check_install_conflicts
from pip._internal.req import install_given_reqs
from pip._internal.req.req_install import (
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.filesystem import test_writable_dir
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import (
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.virtualenv import (
from pip._internal.wheel_builder import build, should_build_for_install_command
def site_packages_writable(root: Optional[str], isolated: bool) -> bool:
    return all((test_writable_dir(d) for d in set(get_lib_location_guesses(root=root, isolated=isolated))))