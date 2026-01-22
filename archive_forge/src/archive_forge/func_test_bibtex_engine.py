from __future__ import unicode_literals
import os
import posixpath
from contextlib import contextmanager
from shutil import rmtree
from tempfile import mkdtemp
import pytest
from pybtex import errors, io
from .utils import diff, get_data
@pytest.mark.parametrize(['filenames'], [(['xampl.bib', 'unsrt.bst'],), (['xampl.bib', 'plain.bst'],), (['xampl.bib', 'alpha.bst'],), (['xampl.bib', 'jurabib.bst'],), (['cyrillic.bib', 'unsrt.bst'],), (['cyrillic.bib', 'alpha.bst'],), (['xampl_mixed.bib', 'unsrt_mixed.bst', 'xampl_mixed_unsrt_mixed.aux'],), (['IEEEtran.bib', 'IEEEtran.bst', 'IEEEtran.aux'],)])
@pytest.mark.parametrize(['check'], [(check_make_bibliography,), (check_format_from_string,)])
def test_bibtex_engine(check, filenames):
    from pybtex import bibtex
    check(bibtex, filenames)