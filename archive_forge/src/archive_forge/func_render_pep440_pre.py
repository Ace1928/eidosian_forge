import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
def render_pep440_pre(pieces):
    """TAG[.postN.devDISTANCE] -- No -dirty.

    Exceptions:
    1: no tags. 0.post0.devDISTANCE
    """
    if pieces['closest-tag']:
        if pieces['distance']:
            tag_version, post_version = pep440_split_post(pieces['closest-tag'])
            rendered = tag_version
            if post_version is not None:
                rendered += f'.post{post_version + 1}.dev{pieces['distance']}'
            else:
                rendered += f'.post0.dev{pieces['distance']}'
        else:
            rendered = pieces['closest-tag']
    else:
        rendered = f'0.post0.dev{pieces['distance']}'
    return rendered