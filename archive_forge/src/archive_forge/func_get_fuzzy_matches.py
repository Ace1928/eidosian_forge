import inspect
import locale
import logging
import logging.handlers
import os
import sys
from cliff import _argparse
from . import complete
from . import help
from . import utils
def get_fuzzy_matches(self, cmd):
    """return fuzzy matches of unknown command
        """
    sep = '_'
    if self.command_manager.convert_underscores:
        sep = ' '
    all_cmds = [k[0] for k in self.command_manager]
    dist = []
    for candidate in sorted(all_cmds):
        prefix = candidate.split(sep)[0]
        if candidate.startswith(cmd):
            dist.append((0, candidate))
            continue
        dist.append((utils.damerau_levenshtein(cmd, prefix, utils.COST) + 1, candidate))
    matches = []
    match_distance = 0
    for distance, candidate in sorted(dist):
        if distance > match_distance:
            if match_distance:
                break
            match_distance = distance
        matches.append(candidate)
    return matches