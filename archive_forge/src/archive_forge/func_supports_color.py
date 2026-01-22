import functools
import os
import sys
from django.utils import termcolors
def supports_color():
    """
    Return True if the running system's terminal supports color,
    and False otherwise.
    """

    def vt_codes_enabled_in_windows_registry():
        """
        Check the Windows Registry to see if VT code handling has been enabled
        by default, see https://superuser.com/a/1300251/447564.
        """
        try:
            import winreg
        except ImportError:
            return False
        else:
            try:
                reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Console')
                reg_key_value, _ = winreg.QueryValueEx(reg_key, 'VirtualTerminalLevel')
            except FileNotFoundError:
                return False
            else:
                return reg_key_value == 1
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    return is_a_tty and (sys.platform != 'win32' or (HAS_COLORAMA and getattr(colorama, 'fixed_windows_console', False)) or 'ANSICON' in os.environ or ('WT_SESSION' in os.environ) or (os.environ.get('TERM_PROGRAM') == 'vscode') or vt_codes_enabled_in_windows_registry())