import sys
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql.language import yaqltypes
def translate_args(without_kwargs, args, kwargs):
    if without_kwargs:
        if len(kwargs) > 0:
            raise exceptions.ArgumentException(next(iter(kwargs)))
        return (args, {})
    pos_args = []
    kw_args = {}
    for t in args:
        if isinstance(t, expressions.MappingRuleExpression):
            param_name = t.source
            if isinstance(param_name, expressions.KeywordConstant):
                param_name = param_name.value
            else:
                raise exceptions.MappingTranslationException()
            kw_args[param_name] = t.destination
        else:
            pos_args.append(t)
    for key, value in kwargs.items():
        if key in kw_args:
            raise exceptions.MappingTranslationException()
        else:
            kw_args[key] = value
    return (tuple(pos_args), kw_args)