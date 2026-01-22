from collections.abc import MutableMapping, MutableSequence
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Type, Union, Tuple
from ..aliases import NamespacesType, BaseXsdType
from .default import ElementData, XMLSchemaConverter
@property
def lossy(self) -> bool:
    return False