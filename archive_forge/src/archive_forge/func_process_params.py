from inspect import Parameter
from parso import tree
from jedi.inference.utils import to_list
from jedi.inference.names import ParamNameWrapper
from jedi.inference.helpers import is_big_annoying_library
@to_list
def process_params(param_names, star_count=3):
    if param_names:
        if is_big_annoying_library(param_names[0].parent_context):
            yield from param_names
            return
    used_names = set()
    arg_callables = []
    kwarg_callables = []
    kw_only_names = []
    kwarg_names = []
    arg_names = []
    original_arg_name = None
    original_kwarg_name = None
    for p in param_names:
        kind = p.get_kind()
        if kind == Parameter.VAR_POSITIONAL:
            if star_count & 1:
                arg_callables = _iter_nodes_for_param(p)
                original_arg_name = p
        elif p.get_kind() == Parameter.VAR_KEYWORD:
            if star_count & 2:
                kwarg_callables = list(_iter_nodes_for_param(p))
                original_kwarg_name = p
        elif kind == Parameter.KEYWORD_ONLY:
            if star_count & 2:
                kw_only_names.append(p)
        elif kind == Parameter.POSITIONAL_ONLY:
            if star_count & 1:
                yield p
        elif star_count == 1:
            yield ParamNameFixedKind(p, Parameter.POSITIONAL_ONLY)
        elif star_count == 2:
            kw_only_names.append(ParamNameFixedKind(p, Parameter.KEYWORD_ONLY))
        else:
            used_names.add(p.string_name)
            yield p
    longest_param_names = ()
    found_arg_signature = False
    found_kwarg_signature = False
    for func_and_argument in arg_callables:
        func, arguments = func_and_argument
        new_star_count = star_count
        if func_and_argument in kwarg_callables:
            kwarg_callables.remove(func_and_argument)
        else:
            new_star_count = 1
        for signature in func.get_signatures():
            found_arg_signature = True
            if new_star_count == 3:
                found_kwarg_signature = True
            args_for_this_func = []
            for p in process_params(list(_remove_given_params(arguments, signature.get_param_names(resolve_stars=False))), new_star_count):
                if p.get_kind() == Parameter.VAR_KEYWORD:
                    kwarg_names.append(p)
                elif p.get_kind() == Parameter.VAR_POSITIONAL:
                    arg_names.append(p)
                elif p.get_kind() == Parameter.KEYWORD_ONLY:
                    kw_only_names.append(p)
                else:
                    args_for_this_func.append(p)
            if len(args_for_this_func) > len(longest_param_names):
                longest_param_names = args_for_this_func
    for p in longest_param_names:
        if star_count == 1 and p.get_kind() != Parameter.VAR_POSITIONAL:
            yield ParamNameFixedKind(p, Parameter.POSITIONAL_ONLY)
        else:
            if p.get_kind() == Parameter.POSITIONAL_OR_KEYWORD:
                used_names.add(p.string_name)
            yield p
    if not found_arg_signature and original_arg_name is not None:
        yield original_arg_name
    elif arg_names:
        yield arg_names[0]
    for func, arguments in kwarg_callables:
        for signature in func.get_signatures():
            found_kwarg_signature = True
            for p in process_params(list(_remove_given_params(arguments, signature.get_param_names(resolve_stars=False))), star_count=2):
                if p.get_kind() == Parameter.VAR_KEYWORD:
                    kwarg_names.append(p)
                elif p.get_kind() == Parameter.KEYWORD_ONLY:
                    kw_only_names.append(p)
    for p in kw_only_names:
        if p.string_name in used_names:
            continue
        yield p
        used_names.add(p.string_name)
    if not found_kwarg_signature and original_kwarg_name is not None:
        yield original_kwarg_name
    elif kwarg_names:
        yield kwarg_names[0]