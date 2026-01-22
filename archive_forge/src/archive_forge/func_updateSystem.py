from .. import functions as fn
from .parameterTypes import GroupParameter
from .SystemSolver import SystemSolver
def updateSystem(self, param, changes):
    changes = [ch for ch in changes if ch[0] not in self._ignoreChange]
    sets = [ch[0] for ch in changes if ch[1] == 'value']
    for param in sets:
        if param in self._fixParams:
            parent = param.parent()
            setattr(self._system, parent.name(), parent.value() if parent.hasValue() else None)
        else:
            setattr(self._system, param.name(), param.value())
    self.updateAllParams()