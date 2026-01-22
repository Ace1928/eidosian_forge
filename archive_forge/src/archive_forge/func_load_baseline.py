import os
import pyomo.core.expr as EXPR
def load_baseline(baseline, testfile, extension, version):
    with open(testfile, 'r') as FILE:
        test = FILE.read()
    if baseline.endswith(f'.{extension}'):
        _tmp = [baseline[:-3]]
    else:
        _tmp = baseline.split(f'.{extension}.', 1)
    _tmp.insert(1, f'expr{int(EXPR.Mode.CURRENT)}')
    _tmp.insert(2, version)
    if not os.path.exists('.'.join(_tmp)):
        _tmp.pop(1)
        if not os.path.exists('.'.join(_tmp)):
            _tmp = []
    if _tmp:
        baseline = '.'.join(_tmp)
    with open(baseline, 'r') as FILE:
        base = FILE.read()
    return (base, test, baseline, testfile)