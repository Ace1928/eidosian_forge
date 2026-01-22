import warnings
from collections.abc import MutableMapping
from copy import copy as _copy
from typing import TYPE_CHECKING, cast, overload, Any, Iterable, Iterator, \
from xml.etree import ElementTree
from .. import limits
from ..exceptions import XMLSchemaValueError
from ..names import XSD_GROUP, XSD_SEQUENCE, XSD_ALL, XSD_CHOICE, XSD_ELEMENT, \
from ..aliases import ElementType, NamespacesType, SchemaType, IterDecodeType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, raw_xml_encode
from ..converters import ElementData
from .exceptions import XMLSchemaModelError, XMLSchemaModelDepthError, \
from .xsdbase import ValidationMixin, XsdComponent, XsdType
from .particles import ParticleMixin, OccursCalculator
from .elements import XsdElement, XsdAlternative
from .wildcards import XsdAnyElement, Xsd11AnyElement
from .models import ModelVisitor, iter_unordered_content, iter_collapsed_content
def get_subgroups(self, item: ModelParticleType) -> List['XsdGroup']:
    """
        Returns a list of the groups that represent the path to the enclosed particle.
        Raises an `XMLSchemaModelError` if *item* is not a particle of the model group.
        """
    subgroups: List[Tuple[XsdGroup, Iterator[ModelParticleType]]] = []
    group, children = (self, iter(self))
    while True:
        for child in children:
            if child is item:
                _subgroups = [x[0] for x in subgroups]
                _subgroups.append(group)
                return _subgroups
            elif isinstance(child, XsdGroup):
                if len(subgroups) > limits.MAX_MODEL_DEPTH:
                    raise XMLSchemaModelDepthError(self)
                subgroups.append((group, children))
                group, children = (child, iter(child))
                break
        else:
            try:
                group, children = subgroups.pop()
            except IndexError:
                msg = _('{!r} is not a particle of the model group')
                raise XMLSchemaModelError(self, msg.format(item)) from None