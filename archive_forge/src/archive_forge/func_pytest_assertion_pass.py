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
def pytest_assertion_pass(item: 'Item', lineno: int, orig: str, expl: str) -> None:
    """Called whenever an assertion passes.

    .. versionadded:: 5.0

    Use this hook to do some processing after a passing assertion.
    The original assertion information is available in the `orig` string
    and the pytest introspected assertion information is available in the
    `expl` string.

    This hook must be explicitly enabled by the ``enable_assertion_pass_hook``
    ini-file option:

    .. code-block:: ini

        [pytest]
        enable_assertion_pass_hook=true

    You need to **clean the .pyc** files in your project directory and interpreter libraries
    when enabling this option, as assertions will require to be re-written.

    :param item: pytest item object of current test.
    :param lineno: Line number of the assert statement.
    :param orig: String with the original assertion.
    :param expl: String with the assert explanation.

    Use in conftest plugins
    =======================

    Any conftest file can implement this hook. For a given item, only conftest
    files in the item's directory and its parent directories are consulted.
    """