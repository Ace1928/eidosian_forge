import signal
from .utils import TimeoutException, BaseTimeout, base_timeoutable
def setup_interrupt(self):
    signal.signal(signal.SIGALRM, self.handle_timeout)
    signal.alarm(self.seconds)