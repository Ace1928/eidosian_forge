import os
import signal
def status_msg(status):
    """Given 'status', which is a process status in the form reported by
    waitpid(2) and returned by process_status(), returns a string describing
    how the process terminated."""
    if os.WIFEXITED(status):
        s = 'exit status %d' % os.WEXITSTATUS(status)
    elif os.WIFSIGNALED(status):
        s = _signal_status_msg('killed', os.WTERMSIG(status))
    elif os.WIFSTOPPED(status):
        s = _signal_status_msg('stopped', os.WSTOPSIG(status))
    else:
        s = 'terminated abnormally (%x)' % status
    if os.WCOREDUMP(status):
        s += ', core dumped'
    return s