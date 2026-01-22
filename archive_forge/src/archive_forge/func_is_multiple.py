from typing import Any, Optional, Tuple, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import ElementType, ModelParticleType
from ..translation import gettext as _
def is_multiple(self) -> bool:
    """Tests the particle can have multiple occurrences."""
    return not self.is_empty() and (not self.is_single())