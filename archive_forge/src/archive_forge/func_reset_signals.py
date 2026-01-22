import os
import signal
import sys
import pickle
from .exceptions import RestartFreqExceeded
from time import monotonic
from io import BytesIO
def reset_signals(handler=_shutdown_cleanup, full=False):
    for sig in TERMSIGS_FULL if full else TERMSIGS_DEFAULT:
        num = signum(sig)
        if num:
            if _should_override_term_signal(sig, signal.getsignal(num)):
                maybe_setsignal(num, handler)
    for sig in TERMSIGS_IGNORE:
        num = signum(sig)
        if num:
            maybe_setsignal(num, signal.SIG_IGN)