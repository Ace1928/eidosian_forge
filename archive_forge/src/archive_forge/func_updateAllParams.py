from .. import functions as fn
from .parameterTypes import GroupParameter
from .SystemSolver import SystemSolver
def updateAllParams(self):
    try:
        self.sigTreeStateChanged.disconnect(self.updateSystem)
        for name, state in self._system._vars.items():
            param = self.child(name)
            try:
                v = getattr(self._system, name)
                if self._system._vars[name][2] is None:
                    self.updateParamState(self.child(name), 'autoSet')
                    param.setValue(v)
                else:
                    self.updateParamState(self.child(name), 'fixed')
            except RuntimeError:
                self.updateParamState(param, 'autoUnset')
    finally:
        self.sigTreeStateChanged.connect(self.updateSystem)