from __future__ import annotations
import math
import os
import re
import textwrap
import warnings
from collections import defaultdict, deque
from functools import partial
from inspect import getfullargspec
from io import StringIO
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.serialization import loadfn
from pymatgen.core import Composition, DummySpecies, Element, Lattice, PeriodicSite, Species, Structure, get_el_sp
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations
from pymatgen.symmetry.groups import SYMM_DATA, SpaceGroup
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list_pbc, in_coord_list_pbc
def get_bibtex_string(self) -> str:
    """
        Get BibTeX reference from CIF file.

        args:
            data:

        Returns:
            BibTeX string.
        """
    try:
        from pybtex.database import BibliographyData, Entry
    except ImportError:
        raise RuntimeError('Bibliographic data extraction requires pybtex.')
    bibtex_keys = {'author': ('_publ_author_name', '_citation_author_name'), 'title': ('_publ_section_title', '_citation_title'), 'journal': ('_journal_name_full', '_journal_name_abbrev', '_citation_journal_full', '_citation_journal_abbrev'), 'volume': ('_journal_volume', '_citation_journal_volume'), 'year': ('_journal_year', '_citation_year'), 'number': ('_journal_number', '_citation_number'), 'page_first': ('_journal_page_first', '_citation_page_first'), 'page_last': ('_journal_page_last', '_citation_page_last'), 'doi': ('_journal_DOI', '_citation_DOI')}
    entries = {}
    for idx, data in enumerate(self._cif.data.values()):
        data = {k.lower(): v for k, v in data.data.items()}
        bibtex_entry = {}
        for field, tags in bibtex_keys.items():
            for tag in tags:
                if tag in data:
                    if isinstance(data[tag], list):
                        bibtex_entry[field] = data[tag][0]
                    else:
                        bibtex_entry[field] = data[tag]
        if 'author' in bibtex_entry:
            if isinstance(bibtex_entry['author'], str) and ';' in bibtex_entry['author']:
                bibtex_entry['author'] = bibtex_entry['author'].split(';')
            if isinstance(bibtex_entry['author'], list):
                bibtex_entry['author'] = ' and '.join(bibtex_entry['author'])
        if 'page_first' in bibtex_entry or 'page_last' in bibtex_entry:
            bibtex_entry['pages'] = bibtex_entry.get('page_first', '') + '--' + bibtex_entry.get('page_last', '')
            bibtex_entry.pop('page_first', None)
            bibtex_entry.pop('page_last', None)
        entries[f'cifref{idx}'] = Entry('article', list(bibtex_entry.items()))
    return BibliographyData(entries).to_string(bib_format='bibtex')