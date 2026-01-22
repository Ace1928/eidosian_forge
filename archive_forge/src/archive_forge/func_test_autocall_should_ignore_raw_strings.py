from IPython.core.splitinput import LineInfo
from IPython.core.prefilter import AutocallChecker
def test_autocall_should_ignore_raw_strings():
    line_info = LineInfo("r'a'")
    pm = ip.prefilter_manager
    ac = AutocallChecker(shell=pm.shell, prefilter_manager=pm, config=pm.config)
    assert ac.check(line_info) is None