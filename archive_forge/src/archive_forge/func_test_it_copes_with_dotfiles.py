from collections.abc import Mapping
from pathlib import Path
import pytest
from jsonschema_specifications import REGISTRY
@pytest.mark.parametrize('ignored_relative_path', ['schemas/.DS_Store', 'schemas/draft7/.DS_Store'])
def test_it_copes_with_dotfiles(ignored_relative_path):
    """
    Ignore files like .DS_Store if someone has actually caused one to exist.

    We test here through the private interface as of course the global has
    already loaded our schemas.
    """
    import jsonschema_specifications
    package = Path(jsonschema_specifications.__file__).parent
    ignored = package / ignored_relative_path
    ignored.touch()
    try:
        list(jsonschema_specifications._schemas())
    finally:
        ignored.unlink()