import sys
import time
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from . import config, errors, osutils
from .repository import _strip_NULL_ghosts
from .revision import CURRENT_REVISION, Revision
def reannotate(parents_lines, new_lines, new_revision_id, _left_matching_blocks=None, heads_provider=None):
    """Create a new annotated version from new lines and parent annotations.

    :param parents_lines: List of annotated lines for all parents
    :param new_lines: The un-annotated new lines
    :param new_revision_id: The revision-id to associate with new lines
        (will often be CURRENT_REVISION)
    :param left_matching_blocks: a hint about which areas are common
        between the text and its left-hand-parent.  The format is
        the SequenceMatcher.get_matching_blocks format
        (start_left, start_right, length_of_match).
    :param heads_provider: An object which provides a .heads() call to resolve
        if any revision ids are children of others.
        If None, then any ancestry disputes will be resolved with
        new_revision_id
    """
    if len(parents_lines) == 0:
        lines = [(new_revision_id, line) for line in new_lines]
    elif len(parents_lines) == 1:
        lines = _reannotate(parents_lines[0], new_lines, new_revision_id, _left_matching_blocks)
    elif len(parents_lines) == 2:
        left = _reannotate(parents_lines[0], new_lines, new_revision_id, _left_matching_blocks)
        lines = _reannotate_annotated(parents_lines[1], new_lines, new_revision_id, left, heads_provider)
    else:
        reannotations = [_reannotate(parents_lines[0], new_lines, new_revision_id, _left_matching_blocks)]
        reannotations.extend((_reannotate(p, new_lines, new_revision_id) for p in parents_lines[1:]))
        lines = []
        for annos in zip(*reannotations):
            origins = {a for a, l in annos}
            if len(origins) == 1:
                lines.append(annos[0])
            else:
                line = annos[0][1]
                if len(origins) == 2 and new_revision_id in origins:
                    origins.remove(new_revision_id)
                if len(origins) == 1:
                    lines.append((origins.pop(), line))
                else:
                    lines.append((new_revision_id, line))
    return lines