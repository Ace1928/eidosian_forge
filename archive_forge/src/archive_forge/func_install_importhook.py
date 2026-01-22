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
def install_importhook(config: Config) -> rewrite.AssertionRewritingHook:
    """Try to install the rewrite hook, raise SystemError if it fails."""
    config.stash[assertstate_key] = AssertionState(config, 'rewrite')
    config.stash[assertstate_key].hook = hook = rewrite.AssertionRewritingHook(config)
    sys.meta_path.insert(0, hook)
    config.stash[assertstate_key].trace('installed rewrite import hook')

    def undo() -> None:
        hook = config.stash[assertstate_key].hook
        if hook is not None and hook in sys.meta_path:
            sys.meta_path.remove(hook)
    config.add_cleanup(undo)
    return hook