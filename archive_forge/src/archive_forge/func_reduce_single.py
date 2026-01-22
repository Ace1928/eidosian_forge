from __future__ import annotations
import re
import typing as T
from . import coredata
from . import mesonlib
from . import mparser
from . import mlog
from .interpreterbase import FeatureNew, FeatureDeprecated, typed_pos_args, typed_kwargs, ContainerTypeInfo, KwargInfo
from .interpreter.type_checking import NoneType, in_set_validator
def reduce_single(self, arg: T.Union[str, mparser.BaseNode]) -> 'TYPE_var':
    if isinstance(arg, str):
        return arg
    if isinstance(arg, mparser.ParenthesizedNode):
        return self.reduce_single(arg.inner)
    elif isinstance(arg, (mparser.BaseStringNode, mparser.BooleanNode, mparser.NumberNode)):
        return arg.value
    elif isinstance(arg, mparser.ArrayNode):
        return [self.reduce_single(curarg) for curarg in arg.args.arguments]
    elif isinstance(arg, mparser.DictNode):
        d = {}
        for k, v in arg.args.kwargs.items():
            if not isinstance(k, mparser.BaseStringNode):
                raise OptionException('Dictionary keys must be a string literal')
            d[k.value] = self.reduce_single(v)
        return d
    elif isinstance(arg, mparser.UMinusNode):
        res = self.reduce_single(arg.value)
        if not isinstance(res, (int, float)):
            raise OptionException('Token after "-" is not a number')
        FeatureNew.single_use('negative numbers in meson_options.txt', '0.54.1', self.subproject)
        return -res
    elif isinstance(arg, mparser.NotNode):
        res = self.reduce_single(arg.value)
        if not isinstance(res, bool):
            raise OptionException('Token after "not" is not a a boolean')
        FeatureNew.single_use('negation ("not") in meson_options.txt', '0.54.1', self.subproject)
        return not res
    elif isinstance(arg, mparser.ArithmeticNode):
        l = self.reduce_single(arg.left)
        r = self.reduce_single(arg.right)
        if not (arg.operation == 'add' and isinstance(l, str) and isinstance(r, str)):
            raise OptionException('Only string concatenation with the "+" operator is allowed')
        FeatureNew.single_use('string concatenation in meson_options.txt', '0.55.0', self.subproject)
        return l + r
    else:
        raise OptionException('Arguments may only be string, int, bool, or array of those.')