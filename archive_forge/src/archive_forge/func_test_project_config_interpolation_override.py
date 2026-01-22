import pytest
import srsly
from confection import ConfigValidationError
from weasel.schemas import ProjectConfigSchema, validate
from weasel.util import is_subpath_of, load_project_config, make_tempdir
from weasel.util import substitute_project_variables, validate_project_commands
@pytest.mark.parametrize('greeting', [342, 'everyone', 'tout le monde', pytest.param('42', marks=pytest.mark.xfail)])
def test_project_config_interpolation_override(greeting):
    variables = {'a': 'world'}
    commands = [{'name': 'x', 'script': ['hello ${vars.a}']}]
    overrides = {'vars.a': greeting}
    project = {'commands': commands, 'vars': variables}
    with make_tempdir() as d:
        srsly.write_yaml(d / 'project.yml', project)
        cfg = load_project_config(d, overrides=overrides)
    assert type(cfg) == dict
    assert type(cfg['commands']) == list
    assert cfg['commands'][0]['script'][0] == f'hello {greeting}'