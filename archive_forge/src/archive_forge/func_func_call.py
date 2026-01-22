import datetime
import warnings
from typing import Any, Literal, Optional, Sequence, Union
from langchain_core.utils import check_package_version
from typing_extensions import TypedDict
from langchain.chains.query_constructor.ir import (
def func_call(self, func_name: Any, args: list) -> FilterDirective:
    func = self._match_func_name(str(func_name))
    if isinstance(func, Comparator):
        if self.allowed_attributes and args[0] not in self.allowed_attributes:
            raise ValueError(f'Received invalid attributes {args[0]}. Allowed attributes are {self.allowed_attributes}')
        return Comparison(comparator=func, attribute=args[0], value=args[1])
    elif len(args) == 1 and func in (Operator.AND, Operator.OR):
        return args[0]
    else:
        return Operation(operator=func, arguments=args)