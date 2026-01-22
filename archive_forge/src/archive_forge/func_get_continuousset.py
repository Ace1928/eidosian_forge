from pyomo.common.deprecation import RenamedClass
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.indexed_component import rule_wrapper
from pyomo.core.base.expression import (
from pyomo.dae.contset import ContinuousSet
from pyomo.dae.diffvar import DAE_Error
def get_continuousset(self):
    """Return the :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>`
        the integral is being taken over
        """
    return self._wrt