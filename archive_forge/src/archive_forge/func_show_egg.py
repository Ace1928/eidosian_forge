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
def show_egg(self, egg_name):
    group_pat = None
    if self.args:
        group_pat = self.get_pattern(self.args[0])
    ep_pat = None
    if len(self.args) > 1:
        ep_pat = self.get_pattern(self.args[1])
    if egg_name.startswith('egg:'):
        egg_name = egg_name[4:]
    dist = pkg_resources.get_distribution(egg_name)
    entry_map = dist.get_entry_map()
    entry_groups = sorted(entry_map.items())
    for group, points in entry_groups:
        if group_pat and (not group_pat.search(group)):
            continue
        print('[%s]' % group)
        points = sorted(points.items())
        for name, entry in points:
            if ep_pat:
                if not ep_pat.search(name):
                    continue
            print(self._ep_description(entry))
            desc = self.get_entry_point_description(entry, group)
            if desc and desc.description:
                print(self.wrap(desc.description, indent=2))
            print()