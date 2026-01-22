import sys
import copy
from pyomo.common.deprecation import deprecation_warning
def main_console_script():
    """This is the entry point for the main Pyomo script"""
    ans = main()
    try:
        return ans.errorcode
    except AttributeError:
        return ans