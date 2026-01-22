import json
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Sequence, Tuple, Type, Union
from .json import pydantic_encoder
from .utils import Representation
def flatten_errors(errors: Sequence[Any], config: Type['BaseConfig'], loc: Optional['Loc']=None) -> Generator['ErrorDict', None, None]:
    for error in errors:
        if isinstance(error, ErrorWrapper):
            if loc:
                error_loc = loc + error.loc_tuple()
            else:
                error_loc = error.loc_tuple()
            if isinstance(error.exc, ValidationError):
                yield from flatten_errors(error.exc.raw_errors, config, error_loc)
            else:
                yield error_dict(error.exc, config, error_loc)
        elif isinstance(error, list):
            yield from flatten_errors(error, config, loc=loc)
        else:
            raise RuntimeError(f'Unknown error object: {error}')