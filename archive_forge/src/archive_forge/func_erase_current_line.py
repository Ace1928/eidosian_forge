import argparse
import os
import platform
import subprocess
import sys
import time
def erase_current_line(self):
    print('\x08' * self.hpos + ' ' * self.hpos + '\x08' * self.hpos, end='')
    sys.stdout.flush()
    self.hpos = 0