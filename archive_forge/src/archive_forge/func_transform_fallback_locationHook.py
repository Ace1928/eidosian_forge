import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
@classmethod
def transform_fallback_locationHook(cls, branch, url):
    """Installed as the 'transform_fallback_location' Branch hook.

        This method calls `transform_fallback_location` on the policy object
        and either returns the url it provides or passes it back to
        check_and_follow_branch_reference.
        """
    try:
        opener = getattr(cls._threading_data, 'opener')
    except AttributeError:
        return url
    new_url, check = opener.policy.transform_fallback_location(branch, url)
    if check:
        return opener.check_and_follow_branch_reference(new_url)
    else:
        return new_url