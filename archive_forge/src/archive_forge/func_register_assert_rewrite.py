import sys
from typing import Any
from typing import Generator
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from _pytest.assertion import rewrite
from _pytest.assertion import truncate
from _pytest.assertion import util
from _pytest.assertion.rewrite import assertstate_key
from _pytest.config import Config
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
def register_assert_rewrite(*names: str) -> None:
    """Register one or more module names to be rewritten on import.

    This function will make sure that this module or all modules inside
    the package will get their assert statements rewritten.
    Thus you should make sure to call this before the module is
    actually imported, usually in your __init__.py if you are a plugin
    using a package.

    :param names: The module names to register.
    """
    for name in names:
        if not isinstance(name, str):
            msg = 'expected module names as *args, got {0} instead'
            raise TypeError(msg.format(repr(names)))
    for hook in sys.meta_path:
        if isinstance(hook, rewrite.AssertionRewritingHook):
            importhook = hook
            break
    else:
        importhook = DummyRewriteHook()
    importhook.mark_rewrite(*names)