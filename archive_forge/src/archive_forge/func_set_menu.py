import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def set_menu(self, option_list):
    """Set the menu option.

        Example set_menu([6,1]) = get all F statistics (menu 6.1)
        """
    self.set_parameter('command', 'MenuOptions=' + '.'.join((str(x) for x in option_list)))