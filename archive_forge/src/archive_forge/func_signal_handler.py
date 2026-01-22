import itertools
from contextlib import ExitStack
def signal_handler(self, signum, frame):
    raise TimeoutException(self.timeout_secs)