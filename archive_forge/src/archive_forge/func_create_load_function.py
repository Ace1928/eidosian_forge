import warnings
from collections import Counter
from functools import lru_cache
from typing import cast, Any, Callable, Dict, List, Iterable, Iterator, \
from ..exceptions import XMLSchemaKeyError, XMLSchemaTypeError, \
from ..names import XSD_NAMESPACE, XSD_REDEFINE, XSD_OVERRIDE, XSD_NOTATION, \
from ..aliases import ComponentClassType, ElementType, SchemaType, BaseXsdType, \
from ..helpers import get_qname, local_name, get_extended_qname
from ..namespaces import NamespaceResourcesMap
from ..translation import gettext as _
from .exceptions import XMLSchemaNotBuiltError, XMLSchemaModelError, XMLSchemaModelDepthError, \
from .xsdbase import XsdValidator, XsdComponent
from .builtins import xsd_builtin_types_factory
from .models import check_model
from . import XsdAttribute, XsdSimpleType, XsdComplexType, XsdElement, XsdAttributeGroup, \
def create_load_function(tag: str) -> Callable[[Dict[str, Any], Iterable[SchemaType]], None]:

    def load_xsd_globals(xsd_globals: Dict[str, Any], schemas: Iterable[SchemaType]) -> None:
        redefinitions = []
        for schema in schemas:
            target_namespace = schema.target_namespace
            for elem in schema.root:
                if elem.tag not in {XSD_REDEFINE, XSD_OVERRIDE}:
                    continue
                location = elem.get('schemaLocation')
                if location is None:
                    continue
                for child in filter(lambda x: x.tag == tag and 'name' in x.attrib, elem):
                    qname = get_qname(target_namespace, child.attrib['name'])
                    redefinitions.append((qname, elem, child, schema, schema.includes[location]))
            for elem in filter(lambda x: x.tag == tag and 'name' in x.attrib, schema.root):
                qname = get_qname(target_namespace, elem.attrib['name'])
                if qname not in xsd_globals:
                    xsd_globals[qname] = (elem, schema)
                else:
                    try:
                        other_schema = xsd_globals[qname][1]
                    except (TypeError, IndexError):
                        pass
                    else:
                        if other_schema.override is schema:
                            continue
                        elif schema.override is other_schema:
                            xsd_globals[qname] = (elem, schema)
                            continue
                    msg = _('global {0} with name={1!r} is already defined')
                    schema.parse_error(msg.format(local_name(tag), qname))
        redefined_names = Counter((x[0] for x in redefinitions))
        for qname, elem, child, schema, redefined_schema in reversed(redefinitions):
            if redefined_names[qname] > 1:
                redefined_names[qname] = 1
                redefined_schemas: Any
                redefined_schemas = [x[-1] for x in redefinitions if x[0] == qname]
                if any((redefined_schemas.count(x) > 1 for x in redefined_schemas)):
                    msg = _('multiple redefinition for {0} {1!r}')
                    schema.parse_error(msg.format(local_name(child.tag), qname), child)
                else:
                    redefined_schemas = {x[-1]: x[-2] for x in redefinitions if x[0] == qname}
                    for rs, s in redefined_schemas.items():
                        while True:
                            try:
                                s = redefined_schemas[s]
                            except KeyError:
                                break
                            if s is rs:
                                msg = _('circular redefinition for {0} {1!r}')
                                schema.parse_error(msg.format(local_name(child.tag), qname), child)
                                break
            if elem.tag == XSD_OVERRIDE:
                if qname in xsd_globals:
                    xsd_globals[qname] = (child, schema)
            else:
                try:
                    xsd_globals[qname].append((child, schema))
                except KeyError:
                    schema.parse_error(_('not a redefinition!'), child)
                except AttributeError:
                    xsd_globals[qname] = [xsd_globals[qname], (child, schema)]
    return load_xsd_globals