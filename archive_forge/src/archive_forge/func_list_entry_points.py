import textwrap
import os
import pkg_resources
from .command import Command, BadCommand
import fnmatch
import re
import traceback
from io import StringIO
import inspect
import types
def list_entry_points(self):
    pattern = self.get_pattern(self.args and self.args[0])
    groups = self.get_groups_by_pattern(pattern)
    print('%i entry point groups found:' % len(groups))
    for group in groups:
        desc = self.get_group_description(group)
        print('[%s]' % group)
        if desc:
            if hasattr(desc, 'description'):
                desc = desc.description
            print(self.wrap(desc, indent=2))