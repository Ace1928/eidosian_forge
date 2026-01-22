import inspect
from yaql.language import exceptions
from yaql.language import utils
from yaql.language import yaqltypes
def convert_arg_func(context2):
    try:
        return param.value_type.convert(val, receiver, context2, self, engine)
    except exceptions.ArgumentValueException:
        raise exceptions.ArgumentException(param.name)