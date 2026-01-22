import os
import time
import pytest
import srsly
from weasel.cli.remote_storage import RemoteStorage
from weasel.schemas import ProjectConfigSchema, validate
from weasel.util import is_subpath_of, load_project_config, make_tempdir
from weasel.util import validate_project_commands
@pytest.mark.parametrize('config', [{'commands': [{'name': 'a'}, {'name': 'a'}]}, {'commands': [{'name': 'a'}], 'workflows': {'a': []}}, {'commands': [{'name': 'a'}], 'workflows': {'b': ['c']}}])
def test_project_config_validation1(config):
    with pytest.raises(SystemExit):
        validate_project_commands(config)