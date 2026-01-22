import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def get_readings(self, agenda):
    readings = []
    agenda_length = len(agenda)
    atomics = dict()
    nonatomics = dict()
    while agenda:
        cur = agenda.pop()
        glue_simp = cur.glue.simplify()
        if isinstance(glue_simp, linearlogic.ImpExpression):
            for key in atomics:
                try:
                    if isinstance(cur.glue, linearlogic.ApplicationExpression):
                        bindings = cur.glue.bindings
                    else:
                        bindings = linearlogic.BindingDict()
                    glue_simp.antecedent.unify(key, bindings)
                    for atomic in atomics[key]:
                        if not cur.indices & atomic.indices:
                            try:
                                agenda.append(cur.applyto(atomic))
                            except linearlogic.LinearLogicApplicationException:
                                pass
                except linearlogic.UnificationException:
                    pass
            try:
                nonatomics[glue_simp.antecedent].append(cur)
            except KeyError:
                nonatomics[glue_simp.antecedent] = [cur]
        else:
            for key in nonatomics:
                for nonatomic in nonatomics[key]:
                    try:
                        if isinstance(nonatomic.glue, linearlogic.ApplicationExpression):
                            bindings = nonatomic.glue.bindings
                        else:
                            bindings = linearlogic.BindingDict()
                        glue_simp.unify(key, bindings)
                        if not cur.indices & nonatomic.indices:
                            try:
                                agenda.append(nonatomic.applyto(cur))
                            except linearlogic.LinearLogicApplicationException:
                                pass
                    except linearlogic.UnificationException:
                        pass
            try:
                atomics[glue_simp].append(cur)
            except KeyError:
                atomics[glue_simp] = [cur]
    for entry in atomics:
        for gf in atomics[entry]:
            if len(gf.indices) == agenda_length:
                self._add_to_reading_list(gf, readings)
    for entry in nonatomics:
        for gf in nonatomics[entry]:
            if len(gf.indices) == agenda_length:
                self._add_to_reading_list(gf, readings)
    return readings