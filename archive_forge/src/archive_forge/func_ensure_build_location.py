import functools
import logging
import os
import shutil
import sys
import uuid
import zipfile
from optparse import Values
from pathlib import Path
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Union
from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.packaging.version import Version
from pip._vendor.packaging.version import parse as parse_version
from pip._vendor.pyproject_hooks import BuildBackendHookCaller
from pip._internal.build_env import BuildEnvironment, NoOpBuildEnvironment
from pip._internal.exceptions import InstallationError, PreviousBuildDirError
from pip._internal.locations import get_scheme
from pip._internal.metadata import (
from pip._internal.metadata.base import FilesystemWheel
from pip._internal.models.direct_url import DirectUrl
from pip._internal.models.link import Link
from pip._internal.operations.build.metadata import generate_metadata
from pip._internal.operations.build.metadata_editable import generate_editable_metadata
from pip._internal.operations.build.metadata_legacy import (
from pip._internal.operations.install.editable_legacy import (
from pip._internal.operations.install.wheel import install_wheel
from pip._internal.pyproject import load_pyproject_toml, make_pyproject_path
from pip._internal.req.req_uninstall import UninstallPathSet
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import (
from pip._internal.utils.packaging import safe_extra
from pip._internal.utils.subprocess import runner_with_spinner_message
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
from pip._internal.utils.unpacking import unpack_file
from pip._internal.utils.virtualenv import running_under_virtualenv
from pip._internal.vcs import vcs
def ensure_build_location(self, build_dir: str, autodelete: bool, parallel_builds: bool) -> str:
    assert build_dir is not None
    if self._temp_build_dir is not None:
        assert self._temp_build_dir.path
        return self._temp_build_dir.path
    if self.req is None:
        self._temp_build_dir = TempDirectory(kind=tempdir_kinds.REQ_BUILD, globally_managed=True)
        return self._temp_build_dir.path
    dir_name: str = canonicalize_name(self.req.name)
    if parallel_builds:
        dir_name = f'{dir_name}_{uuid.uuid4().hex}'
    if not os.path.exists(build_dir):
        logger.debug('Creating directory %s', build_dir)
        os.makedirs(build_dir)
    actual_build_dir = os.path.join(build_dir, dir_name)
    delete_arg = None if autodelete else False
    return TempDirectory(path=actual_build_dir, delete=delete_arg, kind=tempdir_kinds.REQ_BUILD, globally_managed=True).path