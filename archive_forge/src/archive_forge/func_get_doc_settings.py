import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Union
from ..errors import Errors, Warnings
from ..tokens import Doc, Span
from ..util import find_available_port, is_in_jupyter
from .render import DependencyRenderer, EntityRenderer, SpanRenderer
def get_doc_settings(doc: Doc) -> Dict[str, Any]:
    return {'lang': doc.lang_, 'direction': doc.vocab.writing_system.get('direction', 'ltr')}