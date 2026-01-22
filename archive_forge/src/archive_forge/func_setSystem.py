from .. import functions as fn
from .parameterTypes import GroupParameter
from .SystemSolver import SystemSolver
def setSystem(self, sys):
    self._system = sys
    defaults = {}
    vals = {}
    for param in self:
        name = param.name()
        constraints = ''
        if hasattr(sys, '_' + name):
            constraints += 'n'
        if not param.readonly():
            constraints += 'f'
            if 'n' in constraints:
                ch = param.addChild(dict(name='fixed', type='bool', value=False))
                self._fixParams.append(ch)
                param.setReadonly(True)
                param.setOpts(expanded=False)
            else:
                vals[name] = param.value()
                ch = param.addChild(dict(name='fixed', type='bool', value=True, readonly=True))
        defaults[name] = [None, param.type(), None, constraints]
    sys.defaultState.update(defaults)
    sys.reset()
    for name, value in vals.items():
        setattr(sys, name, value)
    self.updateAllParams()