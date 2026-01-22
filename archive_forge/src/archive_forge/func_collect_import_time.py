import re
import sys
import subprocess
import pyomo.common.unittest as unittest
def collect_import_time(module):
    output = subprocess.check_output([sys.executable, '-X', 'importtime', '-c', 'import %s' % (module,)], stderr=subprocess.STDOUT)
    output = output.decode()
    line_re = re.compile('.*:\\s*(\\d+) \\|\\s*(\\d+) \\| ( *)([^ ]+)')
    data = []
    for line in output.splitlines():
        g = line_re.match(line)
        if not g:
            continue
        _self = int(g.group(1))
        _cumul = int(g.group(2))
        _level = len(g.group(3)) // 2
        _module = g.group(4)
        while len(data) < _level + 1:
            data.append(ImportData())
        if len(data) > _level + 1:
            assert len(data) == _level + 2
            inner = data.pop()
            inner.tpl = {(k if '(from' in k else '%s (from %s)' % (k, _module), v) for k, v in inner.tpl.items()}
            if _module.startswith('pyomo'):
                data[_level].update(inner)
                data[_level].pyomo[_module] = _self
            elif _level > 0:
                data[_level].tpl[_module] = _cumul
        elif _module.startswith('pyomo'):
            data[_level].pyomo[_module] = _self
        elif _level > 0:
            data[_level].tpl[_module] = _self
    assert len(data) == 1
    return data[0]