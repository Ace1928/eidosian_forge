import re
from pathlib import Path
def test_build_dependencies():
    libs_ignore_requirements = ['numpy', 'pytest', 'pytest-timeout', 'mock', 'flake8', 'hypothesis', 'pre-commit', 'cython-lint', 'black', 'isort', 'mypy', 'types-dataclasses', 'types-mock', 'types-requests', 'types-setuptools']
    libs_ignore_setup = ['numpy', 'fugashi', 'natto-py', 'pythainlp', 'sudachipy', 'sudachidict_core', 'spacy-pkuseg', 'thinc-apple-ops']
    req_dict = {}
    root_dir = Path(__file__).parent
    req_file = root_dir / 'requirements.txt'
    with req_file.open() as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line.startswith('#'):
                lib, v = _parse_req(line)
                if lib and lib not in libs_ignore_requirements:
                    req_dict[lib] = v
    setup_file = root_dir / 'setup.cfg'
    with setup_file.open() as f:
        lines = f.readlines()
    setup_keys = set()
    for line in lines:
        line = line.strip()
        if not line.startswith('#'):
            lib, v = _parse_req(line)
            if lib and (not lib.startswith('cupy')) and (lib not in libs_ignore_setup):
                req_v = req_dict.get(lib, None)
                assert req_v is not None, '{} in setup.cfg but not in requirements.txt'.format(lib)
                assert lib + v == lib + req_v, '{} has different version in setup.cfg and in requirements.txt: {} and {} respectively'.format(lib, v, req_v)
                setup_keys.add(lib)
    assert sorted(setup_keys) == sorted(req_dict.keys())
    toml_file = root_dir / 'pyproject.toml'
    with toml_file.open() as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().strip(',').strip('"')
        if not line.startswith('#'):
            lib, v = _parse_req(line)
            if lib and lib not in libs_ignore_requirements:
                req_v = req_dict.get(lib, None)
                assert lib + v == lib + req_v, '{} has different version in pyproject.toml and in requirements.txt: {} and {} respectively'.format(lib, v, req_v)