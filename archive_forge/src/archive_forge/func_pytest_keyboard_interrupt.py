from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from pluggy import HookspecMarker
def pytest_keyboard_interrupt(excinfo: 'ExceptionInfo[Union[KeyboardInterrupt, Exit]]') -> None:
    """Called for keyboard interrupt.

    :param excinfo: The exception info.

    Use in conftest plugins
    =======================

    Any conftest plugin can implement this hook.
    """