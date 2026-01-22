import getpass
import sys
def is_interactive():
    """Check if we are in an interractive environment.

    If the rapt token needs refreshing, the user needs to answer the
    challenges.
    If the user is not in an interractive environment, the challenges can not
    be answered and we just wait for timeout for no reason.

    Returns: True if is interactive environment, False otherwise.
    """
    return sys.stdin.isatty()