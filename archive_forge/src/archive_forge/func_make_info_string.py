import random
import sys
from . import Nodes
def make_info_string(data, terminal=False):
    """Create nicely formatted support/branchlengths."""
    if self.plain:
        info_string = ''
    elif self.support_as_branchlengths:
        if terminal:
            info_string = f':{self.max_support:1.2f}'
        elif data.support:
            info_string = f':{data.support:1.2f}'
        else:
            info_string = ':0.00'
    elif self.branchlengths_only:
        info_string = f':{data.branchlength:1.5f}'
    elif terminal:
        info_string = f':{data.branchlength:1.5f}'
    elif data.branchlength is not None and data.support is not None:
        info_string = f'{data.support:1.2f}:{data.branchlength:1.5f}'
    elif data.branchlength is not None:
        info_string = f'0.00000:{data.branchlength:1.5f}'
    elif data.support is not None:
        info_string = f'{data.support:1.2f}:0.00000'
    else:
        info_string = '0.00:0.00000'
    if not ignore_comments:
        try:
            info_string = str(data.nodecomment) + info_string
        except AttributeError:
            pass
    return info_string