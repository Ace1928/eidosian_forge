import inspect
from yaql.language import exceptions
from yaql.language import utils
from yaql.language import yaqltypes
def yaql_property(source_type):

    def decorator(func):

        @name('#property#{0}'.format(get_function_definition(func).name))
        @parameter('obj', source_type)
        def wrapper(obj):
            return func(obj)
        return wrapper
    return decorator