import os
import platform
import re
from pathlib import Path
import pytest
import sklearn
from sklearn._min_dependencies import dependent_packages
from sklearn.utils.fixes import parse_version
def test_min_dependencies_pyproject_toml():
    """Check versions in pyproject.toml is consistent with _min_dependencies."""
    tomllib = pytest.importorskip('tomllib')
    root_directory = Path(sklearn.__file__).parent.parent
    pyproject_toml_path = root_directory / 'pyproject.toml'
    if not pyproject_toml_path.exists():
        pytest.skip('pyproject.toml is not available.')
    with pyproject_toml_path.open('rb') as f:
        pyproject_toml = tomllib.load(f)
    build_requirements = pyproject_toml['build-system']['requires']
    pyproject_build_min_versions = {}
    for requirement in build_requirements:
        if '>=' in requirement:
            if 'numpy>=1.25' in requirement:
                continue
            package, version = requirement.split('>=')
            package = package.lower()
            pyproject_build_min_versions[package] = version
    assert set(['scipy', 'cython']) == set(pyproject_build_min_versions)
    for package, version in pyproject_build_min_versions.items():
        version = parse_version(version)
        expected_min_version = parse_version(dependent_packages[package][0])
        assert version == expected_min_version, f'{package} has a mismatched version'