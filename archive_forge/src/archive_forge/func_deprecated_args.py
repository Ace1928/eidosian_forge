import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
def deprecated_args(date, instructions, *deprecated_arg_names_or_tuples, **kwargs):
    """Decorator for marking specific function arguments as deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called with the deprecated argument. It has the following format:

    Calling <function> (from <module>) with <arg> is deprecated and will be
    removed after <date>. Instructions for updating:
      <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> includes the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated arguments)' is
  appended to the first line of the docstring and a deprecation notice is
  prepended to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed. Must
      be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    *deprecated_arg_names_or_tuples: String or 2-Tuple (String, ok_val).  The
      string is the deprecated argument name. Optionally, an ok-value may be
      provided.  If the user provided argument equals this value, the warning is
      suppressed.
    **kwargs: If `warn_once=False` is passed, every call with a deprecated
      argument will log a warning. The default behavior is to only warn the
      first time the function is called with any given deprecated argument. All
      other kwargs raise `ValueError`.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, instructions are
      empty, the deprecated arguments are not present in the function
      signature, the second element of a deprecated_tuple is not a
      list, or if a kwarg other than `warn_once` is passed.
  """
    _validate_deprecation_args(date, instructions)
    if not deprecated_arg_names_or_tuples:
        raise ValueError('Specify which argument is deprecated.')
    if kwargs and list(kwargs.keys()) != ['warn_once']:
        kwargs.pop('warn_once', None)
        raise ValueError(f'Illegal argument passed to deprecated_args: {kwargs}')
    warn_once = kwargs.get('warn_once', True)

    def _get_arg_names_to_ok_vals():
        """Returns a dict mapping arg_name to DeprecatedArgSpec w/o position."""
        d = {}
        for name_or_tuple in deprecated_arg_names_or_tuples:
            if isinstance(name_or_tuple, tuple):
                d[name_or_tuple[0]] = DeprecatedArgSpec(-1, True, name_or_tuple[1])
            else:
                d[name_or_tuple] = DeprecatedArgSpec(-1, False, None)
        return d

    def _get_deprecated_positional_arguments(names_to_ok_vals, arg_spec):
        """Builds a dictionary from deprecated arguments to their spec.

    Returned dict is keyed by argument name.
    Each value is a DeprecatedArgSpec with the following fields:
       position: The zero-based argument position of the argument
         within the signature.  None if the argument isn't found in
         the signature.
       ok_values:  Values of this argument for which warning will be
         suppressed.

    Args:
      names_to_ok_vals: dict from string arg_name to a list of values, possibly
        empty, which should not elicit a warning.
      arg_spec: Output from tf_inspect.getfullargspec on the called function.

    Returns:
      Dictionary from arg_name to DeprecatedArgSpec.
    """
        arg_space = arg_spec.args + arg_spec.kwonlyargs
        arg_name_to_pos = {name: pos for pos, name in enumerate(arg_space)}
        deprecated_positional_args = {}
        for arg_name, spec in iter(names_to_ok_vals.items()):
            if arg_name in arg_name_to_pos:
                pos = arg_name_to_pos[arg_name]
                deprecated_positional_args[arg_name] = DeprecatedArgSpec(pos, spec.has_ok_value, spec.ok_value)
        return deprecated_positional_args
    deprecated_arg_names = _get_arg_names_to_ok_vals()

    def deprecated_wrapper(func):
        """Deprecation decorator."""
        decorator_utils.validate_callable(func, 'deprecated_args')
        arg_spec = tf_inspect.getfullargspec(func)
        deprecated_positions = _get_deprecated_positional_arguments(deprecated_arg_names, arg_spec)
        is_varargs_deprecated = arg_spec.varargs in deprecated_arg_names
        is_kwargs_deprecated = arg_spec.varkw in deprecated_arg_names
        if len(deprecated_positions) + is_varargs_deprecated + is_kwargs_deprecated != len(deprecated_arg_names_or_tuples):
            known_args = arg_spec.args + arg_spec.kwonlyargs + [arg_spec.varargs, arg_spec.varkw]
            missing_args = [arg_name for arg_name in deprecated_arg_names if arg_name not in known_args]
            raise ValueError(f'The following deprecated arguments are not present in the function signature: {missing_args}. Expected arguments from the following list: {known_args}.')

        def _same_value(a, b):
            """A comparison operation that works for multiple object types.

      Returns True for two empty lists, two numeric values with the
      same value, etc.

      Returns False for (pd.DataFrame, None), and other pairs which
      should not be considered equivalent.

      Args:
        a: value one of the comparison.
        b: value two of the comparison.

      Returns:
        A boolean indicating whether the two inputs are the same value
        for the purposes of deprecation.
      """
            if a is b:
                return True
            try:
                equality = a == b
                if isinstance(equality, bool):
                    return equality
            except TypeError:
                return False
            return False

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            """Deprecation wrapper."""
            if is_in_graph_mode.IS_IN_GRAPH_MODE() and _PRINT_DEPRECATION_WARNINGS:
                invalid_args = []
                named_args = tf_inspect.getcallargs(func, *args, **kwargs)
                for arg_name, spec in iter(deprecated_positions.items()):
                    if spec.position < len(args) and (not (spec.has_ok_value and _same_value(named_args[arg_name], spec.ok_value))):
                        invalid_args.append(arg_name)
                if is_varargs_deprecated and len(args) > len(arg_spec.args):
                    invalid_args.append(arg_spec.varargs)
                if is_kwargs_deprecated and kwargs:
                    invalid_args.append(arg_spec.varkw)
                for arg_name in deprecated_arg_names:
                    if arg_name in kwargs and (not (deprecated_positions[arg_name].has_ok_value and _same_value(named_args[arg_name], deprecated_positions[arg_name].ok_value))):
                        invalid_args.append(arg_name)
                for arg_name in invalid_args:
                    if (func, arg_name) not in _PRINTED_WARNING:
                        if warn_once:
                            _PRINTED_WARNING[func, arg_name] = True
                        _log_deprecation('From %s: calling %s (from %s) with %s is deprecated and will be removed %s.\nInstructions for updating:\n%s', _call_location(), decorator_utils.get_qualified_name(func), func.__module__, arg_name, 'in a future version' if date is None else 'after %s' % date, instructions)
            return func(*args, **kwargs)
        doc = _add_deprecated_arg_notice_to_docstring(func.__doc__, date, instructions, sorted(deprecated_arg_names.keys()))
        return tf_decorator.make_decorator(func, new_func, 'deprecated', doc)
    return deprecated_wrapper