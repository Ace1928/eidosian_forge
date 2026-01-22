import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def frame_by_name(self, fn_fname, ignorekeys=[], check_cache=True):
    """
        Get the details for the specified Frame using the frame's name.

        Usage examples:

        >>> from nltk.corpus import framenet as fn
        >>> f = fn.frame_by_name('Medical_specialties')
        >>> f.ID
        256
        >>> f.name
        'Medical_specialties'
        >>> f.definition # doctest: +NORMALIZE_WHITESPACE
         "This frame includes words that name medical specialties and is closely related to the
          Medical_professionals frame.  The FE Type characterizing a sub-are in a Specialty may also be
          expressed. 'Ralph practices paediatric oncology.'"

        :param fn_fname: The name of the frame
        :type fn_fname: str
        :param ignorekeys: The keys to ignore. These keys will not be
            included in the output. (optional)
        :type ignorekeys: list(str)
        :return: Information about a frame
        :rtype: dict

        Also see the ``frame()`` function for details about what is
        contained in the dict that is returned.
        """
    if check_cache and fn_fname in self._cached_frames:
        return self._frame_idx[self._cached_frames[fn_fname]]
    elif not self._frame_idx:
        self._buildframeindex()
    locpath = os.path.join(f'{self._root}', self._frame_dir, fn_fname + '.xml')
    try:
        with XMLCorpusView(locpath, 'frame') as view:
            elt = view[0]
    except OSError as e:
        raise FramenetError(f'Unknown frame: {fn_fname}') from e
    fentry = self._handle_frame_elt(elt, ignorekeys)
    assert fentry
    fentry.URL = self._fnweb_url + '/' + self._frame_dir + '/' + fn_fname + '.xml'
    for st in fentry.semTypes:
        if st.rootType.name == 'Lexical_type':
            for lu in fentry.lexUnit.values():
                if not any((x is st for x in lu.semTypes)):
                    lu.semTypes.append(st)
    self._frame_idx[fentry.ID] = fentry
    self._cached_frames[fentry.name] = fentry.ID
    '\n        # now set up callables to resolve the LU pointers lazily.\n        # (could also do this here--caching avoids infinite recursion.)\n        for luName,luinfo in fentry.lexUnit.items():\n            fentry.lexUnit[luName] = (lambda luID: Future(lambda: self.lu(luID)))(luinfo.ID)\n        '
    return fentry