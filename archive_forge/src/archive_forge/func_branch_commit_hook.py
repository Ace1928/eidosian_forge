from ... import version_info  # noqa: F401
from ...config import option_registry
from ...hooks import install_lazy_named_hook
def branch_commit_hook(local, master, old_revno, old_revid, new_revno, new_revid):
    """This is the post_commit hook that runs after commit."""
    from . import emailer
    emailer.EmailSender(master, new_revid, master.get_config_stack(), local_branch=local).send_maybe()