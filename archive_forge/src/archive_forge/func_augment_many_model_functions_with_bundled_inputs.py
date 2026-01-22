from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union, Sequence, Dict, Callable
import textwrap
import torch
from torch._C import TupleType, ListType
from torch.jit._recursive import wrap_cpp_module
def augment_many_model_functions_with_bundled_inputs(model: torch.jit.ScriptModule, inputs: Dict[Callable, Optional[Sequence[Tuple[Any, ...]]]], _receive_inflate_expr: Optional[List[str]]=None, info: Optional[Dict[Callable, List[str]]]=None, skip_size_check=False) -> None:
    """Add bundled sample inputs to a model for an arbitrary list of public functions.

    Models with bundled inputs can be invoked in a uniform manner by
    benchmarking and code coverage tools.

    Augmented models will support the following methods:

        `get_all_bundled_inputs_for_<function_name>() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs_for_foo(): model.foo(*inp)`

        `get_bundled_inputs_functions_and_info() -> Dict[str, Dict[str: List[str]]]`
            Returns a dictionary mapping function names to a metadata dictionary.
            This nested dictionary maps preset strings like:
                'get_inputs_function_name' -> the name of a function attribute in this model that can be
                    run to get back a list of inputs corresponding to that function.
                'info' -> the user provided extra information about the bundled inputs

    If forward has bundled inputs then these following functions are also defined:

        `get_all_bundled_inputs() -> List[Tuple[Any, ...]]`
            Returns a list of tuples suitable for passing to the model like
            `for inp in model.get_all_bundled_inputs(): model(*inp)`

        `get_num_bundled_inputs() -> int`
            Equivalent to `len(model.get_all_bundled_inputs())`,
            but slightly easier to call from C++.

    Inputs can be specified in one of two ways:

      - The model can define `_generate_bundled_inputs_for_<function_name>`.
        If the user chooses this method inputs[<function>] should map to None

      - The `inputs` argument to this function can be a dictionary mapping functions to a
        list of inputs, of the same form that will be returned by get_all_bundled_inputs_for_<function_name>.
        The type of the inputs is List[Tuple[Any, ...]]. The outer list corresponds with a
        list of inputs, the inner tuple is the list of args that together make up one input.
        For inputs of functions that take one arg, this will be a tuple of length one. The Any, ...
        is the actual data that makes up the args, e.g. a tensor.

    Info is an optional parameter that maps functions to a list of strings providing extra information about that
    function's bundled inputs. This could be descriptions, expected outputs, etc.
        - Ex: info={model.forward : ['man eating icecream', 'an airplane', 'a dog']}

    This function will attempt to optimize arguments so that (e.g.)
    arguments like `torch.zeros(1000)` will be represented compactly.
    Only top-level arguments will be optimized.
    Tensors in lists or tuples will not.
    """
    if not isinstance(model, torch.jit.ScriptModule):
        raise Exception('Only ScriptModule is supported.')
    if not inputs:
        raise Exception('Please provide inputs for at least 1 function')
    if hasattr(model, 'get_all_bundled_inputs') or hasattr(model, 'get_bundled_inputs_functions_and_info'):
        raise Exception("Models can only be augmented with bundled inputs once. This Model seems to have already been augmented with bundled inputs. Please start afresh with one that doesn't have bundled inputs.")
    get_bundled_inputs_functions_and_info_template = ''
    for function, input_list in inputs.items():
        if hasattr(function, '__name__'):
            function_name = function.__name__
        elif hasattr(function, 'name'):
            function_name = function.name
        else:
            raise Exception('At least one of your functions has no attribute name please ensure all have one. m.foo.name = "foo"')
        if input_list is not None and (not isinstance(input_list, Sequence)):
            raise TypeError(f'Error inputs for function {function_name} is not a Sequence')
        function_arg_types = [arg.type for arg in function.schema.arguments[1:]]
        deflated_inputs_type: ListType = ListType(TupleType(function_arg_types))
        model._c._register_attribute(f'_bundled_inputs_deflated_{function_name}', deflated_inputs_type, [])
        if hasattr(model, '_generate_bundled_inputs_for_' + function_name):
            if input_list is not None:
                raise Exception('inputs[{name}] is not None, but _generate_bundled_inputs_for_{name} is already defined'.format(name=function_name))
        elif input_list is None or len(input_list) == 0:
            raise Exception('inputs for {name} must be specified if _generate_bundled_inputs_for_{name} is not already defined'.format(name=function_name))
        else:
            deflated_inputs = []
            parts = []
            for inp_idx, args in enumerate(input_list):
                if not isinstance(args, Tuple) and (not isinstance(args, List)):
                    raise TypeError(f'Error bundled input for function {function_name} idx: {inp_idx} is not a Tuple or a List')
                deflated_args = []
                parts.append('(')
                for arg_idx, arg in enumerate(args):
                    inflate_helper_fn_name = _get_inflate_helper_fn_name(arg_idx, inp_idx, function_name)
                    deflated, inflater, helper_definition = _inflate_expr(arg, f'deflated[{inp_idx}][{arg_idx}]', inflate_helper_fn_name, skip_size_check=skip_size_check)
                    deflated_args.append(deflated)
                    parts.append(f'    {inflater},')
                    if helper_definition:
                        model.define(textwrap.dedent(helper_definition))
                deflated_inputs.append(tuple(deflated_args))
                parts.append('),')
            parts.append('')
            expr = '\n'.join(parts)
            if _receive_inflate_expr is not None:
                _receive_inflate_expr.append(expr)
            setattr(model, f'_bundled_inputs_deflated_{function_name}', deflated_inputs)
            definition = textwrap.dedent('\n                def _generate_bundled_inputs_for_{name}(self):\n                    deflated = self._bundled_inputs_deflated_{name}\n                    return [\n                {expr}\n                    ]\n                ').format(expr=expr, name=function_name)
            model.define(definition)
        model.define(textwrap.dedent('\n            def get_all_bundled_inputs_for_{name}(self):\n                all_inputs = self._generate_bundled_inputs_for_{name}()\n                assert all_inputs is not None\n                return all_inputs\n            ').format(name=function_name))
        inputs_info = repr(info[function]) if info and function in info else '[]'
        get_bundled_inputs_functions_and_info_template += f"\n            temp_dict : Dict[str,List[str]] = {{}}\n            info: List[str] = {inputs_info}\n\n            temp_dict['info'] = info\n            temp_dict['get_inputs_function_name'] = ['get_all_bundled_inputs_for_{function_name}']\n            all_inputs['{function_name}'] = temp_dict\n            "
        if function_name == 'forward':
            model.define(textwrap.dedent('\n                def get_all_bundled_inputs(self):\n                    return self.get_all_bundled_inputs_for_forward()\n                '))
            model.define(textwrap.dedent('\n                def get_num_bundled_inputs(self):\n                    return len(self.get_all_bundled_inputs_for_forward())\n                '))
    model.define(textwrap.dedent(f'\n        def get_bundled_inputs_functions_and_info(self):\n            all_inputs : Dict[str, Dict[str,List[str]]] = {{}}\n            {get_bundled_inputs_functions_and_info_template}\n            return all_inputs\n        '))