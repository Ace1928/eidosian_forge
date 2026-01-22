import inspect
from yaql.language import exceptions
from yaql.language import utils
from yaql.language import yaqltypes
def strip_hidden_parameters(self):
    fd = self.clone()
    keys_to_remove = set()
    for k, v in fd.parameters.items():
        if not isinstance(v.value_type, yaqltypes.HiddenParameterType):
            continue
        keys_to_remove.add(k)
        if v.position is not None:
            for v2 in fd.parameters.values():
                if v2.position is not None and v2.position > v.position:
                    v2.position -= 1
    for key in keys_to_remove:
        del fd.parameters[key]
    return fd