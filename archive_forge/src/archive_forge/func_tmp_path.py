import dataclasses
import os
from pathlib import Path
import re
from shutil import rmtree
import tempfile
from typing import Any
from typing import Dict
from typing import final
from typing import Generator
from typing import Literal
from typing import Optional
from typing import Union
from .pathlib import cleanup_dead_symlinks
from .pathlib import LOCK_TIMEOUT
from .pathlib import make_numbered_dir
from .pathlib import make_numbered_dir_with_cleanup
from .pathlib import rm_rf
from _pytest.compat import get_user_id
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Item
from _pytest.reports import TestReport
from _pytest.stash import StashKey
@fixture
def tmp_path(request: FixtureRequest, tmp_path_factory: TempPathFactory) -> Generator[Path, None, None]:
    """Return a temporary directory path object which is unique to each test
    function invocation, created as a sub directory of the base temporary
    directory.

    By default, a new base temporary directory is created each test session,
    and old bases are removed after 3 sessions, to aid in debugging.
    This behavior can be configured with :confval:`tmp_path_retention_count` and
    :confval:`tmp_path_retention_policy`.
    If ``--basetemp`` is used then it is cleared each session. See
    :ref:`temporary directory location and retention`.

    The returned object is a :class:`pathlib.Path` object.
    """
    path = _mk_tmp(request, tmp_path_factory)
    yield path
    tmp_path_factory: TempPathFactory = request.session.config._tmp_path_factory
    policy = tmp_path_factory._retention_policy
    result_dict = request.node.stash[tmppath_result_key]
    if policy == 'failed' and result_dict.get('call', True):
        rmtree(path, ignore_errors=True)
    del request.node.stash[tmppath_result_key]