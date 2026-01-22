from __future__ import annotations
import datetime
import json
import re
import sys
from collections import namedtuple
from io import StringIO
from typing import TYPE_CHECKING
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core.structure import Molecule, Structure
@classmethod
def from_structures(cls, structures: Sequence[Structure], authors: Sequence[dict[str, str]], projects=None, references='', remarks=None, data=None, histories=None, created_at=None) -> list[Self]:
    """A convenience method for getting a list of StructureNL objects by
        specifying structures and metadata separately. Some of the metadata
        is applied to all of the structures for ease of use.

        Args:
            structures: A list of Structure objects
            authors: *List* of {"name":'', "email":''} dicts,
                *list* of Strings as 'John Doe <johndoe@gmail.com>',
                or a single String with commas separating authors
            projects: List of Strings ['Project A', 'Project B']. This
                applies to all structures.
            references: A String in BibTeX format. Again, this applies to all
                structures.
            remarks: List of Strings ['Remark A', 'Remark B']
            data: A list of free form dict. Namespaced at the root level
                with an underscore, e.g. {"_materialsproject":<custom data>}
                . The length of data should be the same as the list of
                structures if not None.
            histories: List of list of dicts - [[{'name':'', 'url':'',
                'description':{}}], ...] The length of histories should be the
                same as the list of structures if not None.
            created_at: A datetime object
        """
    data = [{}] * len(structures) if data is None else data
    histories = [[]] * len(structures) if histories is None else histories
    snl_list = []
    for idx, struct in enumerate(structures):
        snl = cls(struct, authors, projects=projects, references=references, remarks=remarks, data=data[idx], history=histories[idx], created_at=created_at)
        snl_list.append(snl)
    return snl_list