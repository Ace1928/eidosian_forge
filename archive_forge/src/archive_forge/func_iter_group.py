from collections import defaultdict, deque
from typing import Any, Counter, Dict, Iterable, Iterator, List, \
from ..exceptions import XMLSchemaValueError
from ..aliases import ModelGroupType, ModelParticleType, SchemaElementType
from ..translation import gettext as _
from .. import limits
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError
from .wildcards import XsdAnyElement, Xsd11AnyElement
from . import groups
def iter_group(self) -> Iterator[ModelParticleType]:
    """Returns an iterator for the current model group."""
    if self.group.max_occurs == 0:
        return iter(())
    elif self.group.model != 'all':
        return iter(self.group)
    else:
        return (e for e in self.group.iter_elements() if not e.is_over(self.occurs[e]))