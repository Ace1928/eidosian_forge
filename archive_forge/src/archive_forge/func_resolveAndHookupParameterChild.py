import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def resolveAndHookupParameterChild(self, functionGroup, childOpts, interactiveFunction):
    if not functionGroup:
        child = Parameter.create(**childOpts)
    else:
        child = functionGroup.addChild(childOpts, existOk=self.existOk)
    if RunOptions.ON_CHANGED in self.runOptions:
        child.sigValueChanged.connect(interactiveFunction.runFromChangedOrChanging)
    if RunOptions.ON_CHANGING in self.runOptions:
        child.sigValueChanging.connect(interactiveFunction.runFromChangedOrChanging)
    return child