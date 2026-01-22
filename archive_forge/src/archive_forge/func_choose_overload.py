import sys
from yaql.language import exceptions
from yaql.language import expressions
from yaql.language import utils
from yaql.language import yaqltypes
def choose_overload(name, candidates, engine, receiver, context, args, kwargs):

    def raise_ambiguous():
        if receiver is utils.NO_VALUE:
            raise exceptions.AmbiguousFunctionException(name)
        else:
            raise exceptions.AmbiguousMethodException(name, receiver)

    def raise_not_found():
        if receiver is utils.NO_VALUE:
            raise exceptions.NoMatchingFunctionException(name)
        else:
            raise exceptions.NoMatchingMethodException(name, receiver)
    candidates2 = []
    lazy_params = None
    no_kwargs = None
    if receiver is not utils.NO_VALUE:
        args = (receiver,) + args
    for level in candidates:
        new_level = []
        for c in level:
            if no_kwargs is None:
                no_kwargs = c.no_kwargs
                args, kwargs = translate_args(no_kwargs, args, kwargs)
            elif no_kwargs != c.no_kwargs:
                raise_ambiguous()
            mapping = c.map_args(args, kwargs, context, engine)
            if mapping is None:
                continue
            pos, kwd = mapping
            lazy = set()
            for i, pos_arg in enumerate(pos):
                if isinstance(pos_arg.value_type, yaqltypes.LazyParameterType):
                    lazy.add(i)
            for key, value in kwd.items():
                if isinstance(value.value_type, yaqltypes.LazyParameterType):
                    lazy.add(key)
            if lazy_params is None:
                lazy_params = lazy
            elif lazy_params != lazy:
                raise_ambiguous()
            new_level.append((c, mapping))
        if new_level:
            candidates2.append(new_level)
    if len(candidates2) == 0:
        raise_not_found()
    arg_evaluator = lambda i, arg: arg(utils.NO_VALUE, context, engine) if i not in lazy_params and isinstance(arg, expressions.Expression) and (not isinstance(arg, expressions.Constant)) else arg
    args = tuple((arg_evaluator(i, arg) for i, arg in enumerate(args)))
    for key, value in kwargs.items():
        kwargs[key] = arg_evaluator(key, value)
    delegate = None
    winner_mapping = None
    for level in candidates2:
        for c, mapping in level:
            try:
                d = c.get_delegate(receiver, engine, context, args, kwargs)
            except exceptions.ArgumentException:
                pass
            else:
                if delegate is not None:
                    if _is_specialization_of(winner_mapping, mapping):
                        continue
                    elif not _is_specialization_of(mapping, winner_mapping):
                        raise_ambiguous()
                delegate = d
                winner_mapping = mapping
        if delegate is not None:
            break
    if delegate is None:
        raise_not_found()
    return lambda: delegate()