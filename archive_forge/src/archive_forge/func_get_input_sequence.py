import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.model import (
def get_input_sequence():
    input_sequence = mpc.TimeSeriesData({'flow_in[*]': [0.1, 1.0, 0.5, 1.3, 1.0, 0.3]}, [0.0, 2.0, 4.0, 6.0, 8.0, 15.0])
    return mpc.data.convert.series_to_interval(input_sequence)