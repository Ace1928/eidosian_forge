import inspect
import pickle
import platform
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import catalogue
import pytest
from confection import Config, ConfigValidationError
from confection.tests.util import Cat, make_tempdir, my_registry
from confection.util import Generator, partial
@pytest.mark.parametrize('d', ['.', ':'])
def test_config_interpolation_sections(d):
    """Test that config sections are interpolated correctly. The parametrized
    value is the final divider (${a.b} vs. ${a:b}). Both should now work and be
    valid. The double {{ }} in the config strings are required to prevent the
    references from being interpreted as an actual f-string variable.
    """
    c_str = '[a]\nfoo = "hello"\nbar = "world"\n\n[b]\nc = ${a}'
    config = Config().from_str(c_str)
    assert config['b']['c'] == config['a']
    c_str = f'[a]\nfoo = "hello"\n\n[a.x]\ny = ${{a{d}b}}\n\n[a.b]\nc = 1\nd = [10]'
    config = Config().from_str(c_str)
    assert config['a']['x']['y'] == config['a']['b']
    c_str = f'[a]\nx = "string"\ny = 10\n\n[b]\nz = "${{a{d}x}}/${{a{d}y}}"'
    config = Config().from_str(c_str)
    assert config['b']['z'] == 'string/10'
    c_str = f'[a]\nx = ["hello", "world"]\n\n[b]\ny = "result: ${{a{d}x}}"'
    config = Config().from_str(c_str)
    assert config['b']['y'] == 'result: ["hello", "world"]'
    c_str = '[a]\nfoo = "x"\n\n[b]\nbar = ${a}\n\n[c]\nbaz = ${b}'
    config = Config().from_str(c_str)
    assert config['b']['bar'] == config['a']
    assert config['c']['baz'] == config['b']
    c_str = f'[a]\nfoo = "x"\n\n[b]\nbar = ${{a}}\n\n[c]\nbaz = ${{b{d}bar}}'
    config = Config().from_str(c_str)
    assert config['c']['baz'] == config['b']['bar']
    c_str = '[a]\nfoo = "x"\n\n[a.b]\nbar = 100\n\n[c]\nbaz = ${a}'
    config = Config().from_str(c_str)
    assert config['c']['baz'] == config['a']
    c_str = '[a]\nfoo ="x"\n\n[a.b]\nbar = ${a}'
    config = Config().from_str(c_str)
    assert config['a']['b']['bar'] == config['a']
    c_str = f'[a]\nfoo = "x"\n\n[b]\nbar = ${{a}}\n\n[c]\nbaz = ${{b.bar{d}foo}}'
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    c_str = f'[a]\nfoo = ${{b{d}bar}}'
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)
    c_str = '[a]\n\n[a.b]\nfoo = "x: ${c}"\n\n[c]\nbar = 1\nbaz = 2'
    with pytest.raises(ConfigValidationError):
        Config().from_str(c_str)