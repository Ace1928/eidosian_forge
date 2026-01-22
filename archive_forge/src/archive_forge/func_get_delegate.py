import inspect
from yaql.language import exceptions
from yaql.language import utils
from yaql.language import yaqltypes
def get_delegate(self, receiver, engine, context, args, kwargs):

    def checked(val, param):
        if not param.value_type.check(val, context, engine):
            raise exceptions.ArgumentException(param.name)

        def convert_arg_func(context2):
            try:
                return param.value_type.convert(val, receiver, context2, self, engine)
            except exceptions.ArgumentValueException:
                raise exceptions.ArgumentException(param.name)
        return convert_arg_func
    kwargs = kwargs.copy()
    kwargs = dict(kwargs)
    positional = 0
    for arg_name, p in self.parameters.items():
        if p.position is not None and arg_name != '*':
            positional += 1
    positional_args = positional * [None]
    positional_fix_table = positional * [0]
    keyword_args = {}
    for p in self.parameters.values():
        if p.position is not None and isinstance(p.value_type, yaqltypes.HiddenParameterType):
            for index in range(p.position + 1, positional):
                positional_fix_table[index] += 1
    for key, p in self.parameters.items():
        arg_name = p.alias or p.name
        if p.position is not None and key != '*':
            if isinstance(p.value_type, yaqltypes.HiddenParameterType):
                positional_args[p.position] = checked(None, p)
                positional -= 1
            elif p.position - positional_fix_table[p.position] < len(args) and args[p.position - positional_fix_table[p.position]] is not utils.NO_VALUE:
                if arg_name in kwargs:
                    raise exceptions.ArgumentException(p.name)
                positional_args[p.position] = checked(args[p.position - positional_fix_table[p.position]], p)
            elif arg_name in kwargs:
                positional_args[p.position] = checked(kwargs.pop(arg_name), p)
            elif p.default is not NO_DEFAULT:
                positional_args[p.position] = checked(p.default, p)
            else:
                raise exceptions.ArgumentException(p.name)
        elif p.position is None and key != '**':
            if isinstance(p.value_type, yaqltypes.HiddenParameterType):
                keyword_args[key] = checked(None, p)
            elif arg_name in kwargs:
                keyword_args[key] = checked(kwargs.pop(arg_name), p)
            elif p.default is not NO_DEFAULT:
                keyword_args[key] = checked(p.default, p)
            else:
                raise exceptions.ArgumentException(p.name)
    if len(args) > positional:
        if '*' in self.parameters:
            argdef = self.parameters['*']
            positional_args.extend(map(lambda t: checked(t, argdef), args[positional:]))
        else:
            raise exceptions.ArgumentException('*')
    if len(kwargs) > 0:
        if '**' in self.parameters:
            argdef = self.parameters['**']
            for key, value in kwargs.items():
                keyword_args[key] = checked(value, argdef)
        else:
            raise exceptions.ArgumentException('**')

    def func():
        new_context = context.create_child_context()
        result = self.payload(*tuple(map(lambda t: t(new_context), positional_args)), **dict(map(lambda t: (t[0], t[1](new_context)), keyword_args.items())))
        return result
    return func