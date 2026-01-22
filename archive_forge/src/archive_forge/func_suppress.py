from __future__ import unicode_literals
import collections
import logging
from cmakelang.lint import lintdb
def suppress(self, lineno, idlist):
    """
    Given a list of lint ids, enable a suppression for each one which is not
    already supressed. Return the list of new suppressions
    """
    new_suppressions = []
    for idstr in idlist:
        if idstr in self._suppressions:
            continue
        self._suppressions.add(idstr)
        new_suppressions.append(idstr)
    self._suppression_events.append(SuppressionEvent(lineno, 'add', list(new_suppressions)))
    return new_suppressions