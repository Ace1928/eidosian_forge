from collections.abc import MutableMapping, MutableSequence
from typing import TYPE_CHECKING, cast, Any, Dict, Union
from .default import ElementData, XMLSchemaConverter

        Extracts XML decoded data from a data structure for encoding into an ElementTree.

        :param obj: the decoded object.
        :param xsd_element: the `XsdElement` associated to the decoded data structure.
        :param level: the level related to the encoding process (0 means the root).
        :return: an ElementData instance.
        