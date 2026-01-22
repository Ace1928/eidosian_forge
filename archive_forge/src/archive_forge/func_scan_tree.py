import contextlib
from . import errors
from .controldir import ControlDir
from .i18n import gettext
from .trace import note
def scan_tree(base_tree, tree, needed_refs, exit_stack):
    """Scan a tree for refs.

    :param base_tree: The original tree check opened, used to detect duplicate
        tree checks.
    :param tree:  The tree to schedule for checking.
    :param needed_refs: Refs we are accumulating.
    :param exit_stack: The exit stack accumulating.
    """
    if base_tree is not None and tree.basedir == base_tree.basedir:
        return
    note(gettext("Checking working tree at '%s'.") % (tree.basedir,))
    exit_stack.enter_context(tree.lock_read())
    tree_refs = tree._get_check_refs()
    for ref in tree_refs:
        reflist = needed_refs.setdefault(ref, [])
        reflist.append(tree)