from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union
from ._events import (
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import (  # Import the internal things we need
from ._writers import WRITERS, WritersType
@property
def our_state(self) -> Type[Sentinel]:
    """The current state of whichever role we are playing. See
        :ref:`state-machine` for details.
        """
    return self._cstate.states[self.our_role]