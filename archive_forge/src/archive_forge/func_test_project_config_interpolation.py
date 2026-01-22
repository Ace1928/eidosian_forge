import pytest
import srsly
from confection import ConfigValidationError
from weasel.schemas import ProjectConfigSchema, validate
from weasel.util import is_subpath_of, load_project_config, make_tempdir
from weasel.util import substitute_project_variables, validate_project_commands
@pytest.mark.parametrize('int_value', [10, pytest.param('10', marks=pytest.mark.xfail)])
def test_project_config_interpolation(int_value):
    variables = {'a': int_value, 'b': {'c': 'foo', 'd': True}}
    commands = [{'name': 'x', 'script': ['hello ${vars.a} ${vars.b.c}']}, {'name': 'y', 'script': ['${vars.b.c} ${vars.b.d}']}]
    project = {'commands': commands, 'vars': variables}
    with make_tempdir() as d:
        srsly.write_yaml(d / 'project.yml', project)
        cfg = load_project_config(d)
    assert type(cfg) == dict
    assert type(cfg['commands']) == list
    assert cfg['commands'][0]['script'][0] == 'hello 10 foo'
    assert cfg['commands'][1]['script'][0] == 'foo true'
    commands = [{'name': 'x', 'script': ['hello ${vars.a} ${vars.b.e}']}]
    project = {'commands': commands, 'vars': variables}
    with pytest.raises(ConfigValidationError):
        substitute_project_variables(project)