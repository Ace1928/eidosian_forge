import os
import sqlite3
from nltk.corpus.reader.api import CorpusReader
def meanings(self, expr_uid, expr_tt):
    """
        Return a list of meanings for an expression.

        :param expr_uid: the expression's language variety, as a seven-character
            uniform identifier.
        :param expr_tt: the expression's text.
        :return: a list of Meaning objects.
        :rtype: list(Meaning)
        """
    expr_lv = self._uid_lv[expr_uid]
    mn_info = {}
    for i in self._c.execute(self.MEANING_Q, (expr_tt, expr_lv)):
        mn = i[0]
        uid = self._lv_uid[i[5]]
        if not mn in mn_info:
            mn_info[mn] = {'uq': i[1], 'ap': i[2], 'ui': i[3], 'ex': {expr_uid: [expr_tt]}}
        if not uid in mn_info[mn]['ex']:
            mn_info[mn]['ex'][uid] = []
        mn_info[mn]['ex'][uid].append(i[4])
    return [Meaning(mn, mn_info[mn]) for mn in mn_info]