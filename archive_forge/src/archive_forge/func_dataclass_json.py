import abc
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union, overload
from dataclasses_json.cfg import config, LetterCase
from dataclasses_json.core import (Json, _ExtendedEncoder, _asdict,
from dataclasses_json.mm import (JsonData, SchemaType, build_schema)
from dataclasses_json.undefined import Undefined
from dataclasses_json.utils import (_handle_undefined_parameters_safe,
def dataclass_json(_cls: Optional[Type[T]]=None, *, letter_case: Optional[LetterCase]=None, undefined: Optional[Union[str, Undefined]]=None) -> Union[Callable[[Type[T]], Type[T]], Type[T]]:
    """
    Based on the code in the `dataclasses` module to handle optional-parens
    decorators. See example below:

    @dataclass_json
    @dataclass_json(letter_case=LetterCase.CAMEL)
    class Example:
        ...
    """

    def wrap(cls: Type[T]) -> Type[T]:
        return _process_class(cls, letter_case, undefined)
    if _cls is None:
        return wrap
    return wrap(_cls)