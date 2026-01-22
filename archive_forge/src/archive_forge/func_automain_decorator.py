import sys
from .errors import AutocommandError
def automain_decorator(main):
    sys.exit(main(*args, **kwargs))