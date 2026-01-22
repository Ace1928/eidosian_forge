import json
import logging
import os
import re
import shutil
import sys
from base64 import b64encode
from typing import Dict
import requests
from requests.compat import urljoin
import wandb
import wandb.util
from wandb.sdk.lib import filesystem
@magic_arguments()
@argument('path', default=None, nargs='?', help='A path to a resource you want to display, defaults to wandb.run.path')
@argument('-w', '--workspace', default=False, action='store_true', help='Display the entire run project workspace')
@argument('-q', '--quiet', default=False, action='store_true', help='Display the minimal amount of output')
@argument('-h', '--height', default=420, type=int, help='The height of the iframe in pixels')
@line_cell_magic
def wandb(self, line, cell=None):
    """Display wandb resources in jupyter.  This can be used as cell or line magic.

        %wandb USERNAME/PROJECT/runs/RUN_ID
        ---
        %%wandb -h 1024
        with wandb.init() as run:
            run.log({"loss": 1})
        """
    args = parse_argstring(self.wandb, line)
    self.options['height'] = args.height
    self.options['workspace'] = args.workspace
    self.options['quiet'] = args.quiet
    iframe = IFrame(args.path, opts=self.options)
    displayed = iframe.maybe_display()
    if cell is not None:
        if not displayed:
            cell = f'wandb.jupyter.__IFrame = wandb.jupyter.IFrame(opts={self.options})\n' + cell + '\nwandb.jupyter.__IFrame = None'
        get_ipython().run_cell(cell)