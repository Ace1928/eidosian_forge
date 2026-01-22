from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cmd
import shlex
from typing import List, Optional
from absl import flags
from pyglib import appcommands
import bq_utils
from frontend import bigquery_command
from frontend import bq_cached_client
def completedefault(self, unused_text, line: str, unused_begidx, unused_endidx):
    if not line:
        return []
    else:
        command_name = line.partition(' ')[0].lower()
        usage = ''
        if command_name in self._commands:
            usage = self._commands[command_name].usage
        elif command_name == 'set':
            usage = 'set (project_id|dataset_id) <name>'
        elif command_name == 'unset':
            usage = 'unset (project_id|dataset_id)'
        if usage:
            print()
            print(usage)
            print('%s%s' % (self.prompt, line), end=' ')
        return []