import threading
from breezy import strace, tests
from breezy.strace import StraceResult, strace_detailed
from breezy.tests.features import strace_feature
def strace_detailed_or_skip(self, *args, **kwargs):
    """Run strace, but cope if it's not allowed"""
    try:
        return strace_detailed(*args, **kwargs)
    except strace.StraceError as e:
        if e.err_messages.startswith('attach: ptrace(PTRACE_ATTACH, ...): Operation not permitted'):
            raise tests.TestSkipped('ptrace not permitted')
        else:
            raise