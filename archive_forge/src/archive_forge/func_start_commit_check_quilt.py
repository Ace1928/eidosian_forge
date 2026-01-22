from ... import trace
from ...errors import BzrError
from ...hooks import install_lazy_named_hook
from ...config import Option, bool_from_store, option_registry
def start_commit_check_quilt(tree):
    """start_commit hook which checks the state of quilt patches.
    """
    config = tree.get_config_stack()
    policy = config.get('quilt.commit_policy')
    from .merge import start_commit_quilt_patches
    from .wrapper import QuiltNotInstalled
    try:
        start_commit_quilt_patches(tree, policy)
    except QuiltNotInstalled:
        trace.warning('quilt not installed; not updating patches')