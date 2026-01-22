from __future__ import annotations
from shutil import get_terminal_size
def check_main():
    try:
        import __main__ as main
    except ModuleNotFoundError:
        return get_option('mode.sim_interactive')
    return not hasattr(main, '__file__') or get_option('mode.sim_interactive')