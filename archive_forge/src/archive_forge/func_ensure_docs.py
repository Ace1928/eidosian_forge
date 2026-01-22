import re
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union
import srsly
from thinc.api import (
from thinc.config import ConfigValidationError
from wasabi import Printer
from ..errors import Errors
from ..schemas import ConfigSchemaPretrain
from ..tokens import Doc
from ..util import dot_to_object, load_model_from_config, registry
from .example import Example
def ensure_docs(examples_or_docs: Iterable[Union[Doc, Example]]) -> List[Doc]:
    docs = []
    for eg_or_doc in examples_or_docs:
        if isinstance(eg_or_doc, Doc):
            docs.append(eg_or_doc)
        else:
            docs.append(eg_or_doc.reference)
    return docs