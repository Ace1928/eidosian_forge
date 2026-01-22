import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def print_announcements(opt):
    """
    Output any announcements the ParlAI team wishes to make to users.

    Also gives the user the option to suppress the output.
    """
    return
    noannounce_file = os.path.join(opt.get('datapath'), 'noannouncements')
    if os.path.exists(noannounce_file):
        return
    RESET = '\x1b[0m'
    BOLD = '\x1b[1m'
    RED = '\x1b[1;91m'
    YELLOW = '\x1b[1;93m'
    GREEN = '\x1b[1;92m'
    BLUE = '\x1b[1;96m'
    CYAN = '\x1b[1;94m'
    MAGENTA = '\x1b[1;95m'
    USE_COLORS = _sys.stdout.isatty()
    if not USE_COLORS:
        RESET = BOLD = RED = YELLOW = GREEN = BLUE = CYAN = MAGENTA = ''
    rainbow = [RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA]
    size = 78 // len(rainbow)
    stars = ''.join([color + '*' * size for color in rainbow])
    stars += RESET
    print('\n'.join(['', stars, BOLD, 'Announcements go here.', RESET, 'To suppress this message (and future announcements), run\n`touch {}`'.format(noannounce_file), stars]))