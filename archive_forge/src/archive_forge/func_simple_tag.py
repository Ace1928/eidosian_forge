from functools import wraps
from importlib import import_module
from inspect import getfullargspec, unwrap
from django.utils.html import conditional_escape
from django.utils.itercompat import is_iterable
from .base import Node, Template, token_kwargs
from .exceptions import TemplateSyntaxError
def simple_tag(self, func=None, takes_context=None, name=None):
    """
        Register a callable as a compiled template tag. Example:

        @register.simple_tag
        def hello(*args, **kwargs):
            return 'world'
        """

    def dec(func):
        params, varargs, varkw, defaults, kwonly, kwonly_defaults, _ = getfullargspec(unwrap(func))
        function_name = name or func.__name__

        @wraps(func)
        def compile_func(parser, token):
            bits = token.split_contents()[1:]
            target_var = None
            if len(bits) >= 2 and bits[-2] == 'as':
                target_var = bits[-1]
                bits = bits[:-2]
            args, kwargs = parse_bits(parser, bits, params, varargs, varkw, defaults, kwonly, kwonly_defaults, takes_context, function_name)
            return SimpleNode(func, takes_context, args, kwargs, target_var)
        self.tag(function_name, compile_func)
        return func
    if func is None:
        return dec
    elif callable(func):
        return dec(func)
    else:
        raise ValueError('Invalid arguments provided to simple_tag')