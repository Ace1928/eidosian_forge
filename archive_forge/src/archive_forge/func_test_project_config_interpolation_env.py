import pytest
import srsly
from confection import ConfigValidationError
from weasel.schemas import ProjectConfigSchema, validate
from weasel.util import is_subpath_of, load_project_config, make_tempdir
from weasel.util import substitute_project_variables, validate_project_commands
def test_project_config_interpolation_env(monkeypatch: pytest.MonkeyPatch):
    variables = {'a': 10}
    env_var = 'SPACY_TEST_FOO'
    env_vars = {'foo': env_var}
    commands = [{'name': 'x', 'script': ['hello ${vars.a} ${env.foo}']}]
    project = {'commands': commands, 'vars': variables, 'env': env_vars}
    with make_tempdir() as d:
        srsly.write_yaml(d / 'project.yml', project)
        cfg = load_project_config(d)
    assert cfg['commands'][0]['script'][0] == 'hello 10 '
    monkeypatch.setenv(env_var, '123')
    with make_tempdir() as d:
        srsly.write_yaml(d / 'project.yml', project)
        cfg = load_project_config(d)
    assert cfg['commands'][0]['script'][0] == 'hello 10 123'