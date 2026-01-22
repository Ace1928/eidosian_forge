import os
import time
import pytest
import srsly
from weasel.cli.remote_storage import RemoteStorage
from weasel.schemas import ProjectConfigSchema, validate
from weasel.util import is_subpath_of, load_project_config, make_tempdir
from weasel.util import validate_project_commands
def test_issue11235():
    """
    Test that the cli handles interpolation in the directory names correctly when loading project config.
    """
    lang_var = 'en'
    variables = {'lang': lang_var}
    commands = [{'name': 'x', 'script': ['hello ${vars.lang}']}]
    directories = ['cfg', '${vars.lang}_model']
    project = {'commands': commands, 'vars': variables, 'directories': directories}
    with make_tempdir() as d:
        srsly.write_yaml(d / 'project.yml', project)
        cfg = load_project_config(d)
        assert os.path.exists(d / 'cfg')
        assert os.path.exists(d / f'{lang_var}_model')
    assert cfg['commands'][0]['script'][0] == f'hello {lang_var}'