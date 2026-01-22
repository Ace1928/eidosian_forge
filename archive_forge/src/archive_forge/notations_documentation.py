from typing import Optional
from ..names import XSD_NOTATION
from ..translation import gettext as _
from ..helpers import get_qname
from .xsdbase import XsdComponent

    Class for XSD *notation* declarations.

    ..  <notation
          id = ID
          name = NCName
          public = token
          system = anyURI
          {any attributes with non-schema namespace}...>
          Content: (annotation?)
        </notation>
    