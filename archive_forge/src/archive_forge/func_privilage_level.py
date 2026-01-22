from lazyops.types.common import UpperStrEnum
from typing import Union
@property
def privilage_level(self) -> int:
    """
        Returns the privilage level of the user role
        - Can be subclassed to return a different value
        """
    return UserPrivilageLevel[self.value]