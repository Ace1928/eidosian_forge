from IPython.utils.capture import capture_output
import pytest
def test_alias_lifecycle():
    name = 'test_alias1'
    cmd = 'echo "Hello"'
    am = _ip.alias_manager
    am.clear_aliases()
    am.define_alias(name, cmd)
    assert am.is_alias(name)
    assert am.retrieve_alias(name) == cmd
    assert (name, cmd) in am.aliases
    orig_system = _ip.system
    result = []
    _ip.system = result.append
    try:
        _ip.run_cell('%{}'.format(name))
        result = [c.strip() for c in result]
        assert result == [cmd]
    finally:
        _ip.system = orig_system
    am.undefine_alias(name)
    assert not am.is_alias(name)
    with pytest.raises(ValueError):
        am.retrieve_alias(name)
    assert (name, cmd) not in am.aliases