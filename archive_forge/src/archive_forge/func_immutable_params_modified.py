import abc
import collections
import itertools
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
def immutable_params_modified(self, new_parameters, input_params):
    common_params = list(set(new_parameters.non_pseudo_param_keys) & set(self.non_pseudo_param_keys))
    invalid_params = []
    for param in common_params:
        old_value = self.params[param]
        if param in input_params:
            new_value = input_params[param]
        else:
            new_value = new_parameters[param]
        immutable = new_parameters.params[param].schema.immutable
        if immutable and old_value.value() != new_value:
            invalid_params.append(param)
    if invalid_params:
        return invalid_params