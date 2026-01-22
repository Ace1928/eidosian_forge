import os
import time
import pytest
import srsly
from weasel.cli.remote_storage import RemoteStorage
from weasel.schemas import ProjectConfigSchema, validate
from weasel.util import is_subpath_of, load_project_config, make_tempdir
from weasel.util import validate_project_commands
def test_local_remote_storage_pull_missing():
    with make_tempdir() as d:
        filename = 'a.txt'
        remote = RemoteStorage(d / 'root', str(d / 'remote'))
        assert remote.pull(filename, command_hash='aaaa') is None
        assert remote.pull(filename) is None