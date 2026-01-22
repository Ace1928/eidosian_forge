from pyomo.dae.flatten import flatten_dae_components
from pyomo.common.modeling import NOTSET
from pyomo.core.base.var import Var
from pyomo.core.base.expression import Expression
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.expr.numeric_expr import value as pyo_value
from pyomo.contrib.mpc.interfaces.load_data import (
from pyomo.contrib.mpc.interfaces.copy_values import copy_values_at_time
from pyomo.contrib.mpc.data.find_nearest_index import find_nearest_index
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
from pyomo.contrib.mpc.data.dynamic_data_base import _is_iterable
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.convert import _process_to_dynamic_data
from pyomo.contrib.mpc.modeling.cost_expressions import (
from pyomo.contrib.mpc.modeling.constraints import get_piecewise_constant_constraints
def copy_values_at_time(self, source_time=None, target_time=None):
    """
        Copy values of all time-indexed variables from source time point
        to target time points.

        Parameters
        ----------
        source_time: Float
            Time point from which to copy values.
        target_time: Float or iterable
            Time point or points to which to copy values.

        """
    if source_time is None:
        source_time = self.time.first()
    if target_time is None:
        target_time = self.time
    copy_values_at_time(self._dae_vars, self._dae_vars, source_time, target_time)