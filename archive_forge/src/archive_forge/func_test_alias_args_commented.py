from IPython.utils.capture import capture_output
import pytest
def test_alias_args_commented():
    """Check that alias correctly ignores 'commented out' args"""
    _ip.run_line_magic('alias', 'commentarg echo this is %%s a commented out arg')
    with capture_output() as cap:
        _ip.run_cell('commentarg')
    assert cap.stdout.strip() == 'this is %s a commented out arg'