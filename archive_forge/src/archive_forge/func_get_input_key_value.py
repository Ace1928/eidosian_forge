import collections
import copy
import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.heat import software_config as sc
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import support
@staticmethod
def get_input_key_value(fn_arg, inputs, check_input_val='LAX'):
    if check_input_val == 'STRICT' and fn_arg not in inputs:
        raise exception.UserParameterMissing(key=fn_arg)
    return inputs.get(fn_arg)