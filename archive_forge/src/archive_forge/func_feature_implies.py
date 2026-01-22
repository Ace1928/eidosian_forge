import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def feature_implies(self, names, keep_origins=False):
    """
        Return a set of CPU features that implied by 'names'

        Parameters
        ----------
        names : str or sequence of str
            CPU feature name(s) in uppercase.

        keep_origins : bool
            if False(default) then the returned set will not contain any
            features from 'names'. This case happens only when two features
            imply each other.

        Examples
        --------
        >>> self.feature_implies("SSE3")
        {'SSE', 'SSE2'}
        >>> self.feature_implies("SSE2")
        {'SSE'}
        >>> self.feature_implies("SSE2", keep_origins=True)
        # 'SSE2' found here since 'SSE' and 'SSE2' imply each other
        {'SSE', 'SSE2'}
        """

    def get_implies(name, _caller=set()):
        implies = set()
        d = self.feature_supported[name]
        for i in d.get('implies', []):
            implies.add(i)
            if i in _caller:
                continue
            _caller.add(name)
            implies = implies.union(get_implies(i, _caller))
        return implies
    if isinstance(names, str):
        implies = get_implies(names)
        names = [names]
    else:
        assert hasattr(names, '__iter__')
        implies = set()
        for n in names:
            implies = implies.union(get_implies(n))
    if not keep_origins:
        implies.difference_update(names)
    return implies