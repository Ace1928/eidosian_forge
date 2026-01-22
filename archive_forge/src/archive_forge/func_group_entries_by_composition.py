from __future__ import annotations
import collections
import csv
import datetime
import itertools
import json
import logging
import multiprocessing as mp
import re
from typing import TYPE_CHECKING, Literal
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.analysis.structure_matcher import SpeciesComparator, StructureMatcher
from pymatgen.core import Composition, Element
def group_entries_by_composition(entries, sort_by_e_per_atom=True):
    """Given a sequence of Entry-like objects, group them by composition and
        optionally sort by energy above hull.

    Args:
        entries (list): Sequence of Entry-like objects.
        sort_by_e_per_atom (bool): Whether to sort the grouped entries by
            energy per atom (lowest energy first). Default True.

    Returns:
        Sequence of sequence of entries by composition. e.g,
        [[ entry1, entry2], [entry3, entry4, entry5]]
    """
    entry_groups = []
    entries = sorted(entries, key=lambda e: e.reduced_formula)
    for _, g in itertools.groupby(entries, key=lambda e: e.reduced_formula):
        group = list(g)
        if sort_by_e_per_atom:
            group = sorted(group, key=lambda e: e.energy_per_atom)
        entry_groups.append(group)
    return entry_groups