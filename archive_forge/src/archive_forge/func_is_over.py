from typing import Any, Optional, Tuple, Union
from ..exceptions import XMLSchemaValueError
from ..aliases import ElementType, ModelParticleType
from ..translation import gettext as _
def is_over(self, occurs: int) -> bool:
    """Tests if provided occurrences are over the maximum."""
    return self.max_occurs is not None and self.max_occurs <= occurs