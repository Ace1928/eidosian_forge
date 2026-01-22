from __future__ import annotations
import ast
import functools
import json
import re
import warnings
from collections import Counter
from enum import Enum, unique
from itertools import combinations, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core.units import SUPPORTED_UNIT_NAMES, FloatWithUnit, Ha_to_eV, Length, Mass, Unit
from pymatgen.util.string import Stringify, formula_double_format
@property
def term_symbols(self) -> list[list[str]]:
    """All possible Russell-Saunders term symbol of the Element.
        eg. L = 1, n_e = 2 (s2) returns [['1D2'], ['3P0', '3P1', '3P2'], ['1S0']].
        """
    if self.is_noble_gas:
        return [['1S0']]
    L_symbols = 'SPDFGHIKLMNOQRTUVWXYZ'
    L, v_e = self.valence
    ml = list(range(-L, L + 1))
    ms = [1 / 2, -1 / 2]
    ml_ms = list(product(ml, ms))
    n = (2 * L + 1) * 2
    e_config_combs = list(combinations(range(n), v_e))
    TL = [sum((ml_ms[comb[e]][0] for e in range(v_e))) for comb in e_config_combs]
    TS = [sum((ml_ms[comb[e]][1] for e in range(v_e))) for comb in e_config_combs]
    comb_counter = Counter(zip(TL, TS))
    term_symbols = []
    while sum(comb_counter.values()) > 0:
        L, S = min(comb_counter)
        J = list(np.arange(abs(L - S), abs(L) + abs(S) + 1))
        term_symbols.append([f'{int(2 * abs(S) + 1)}{L_symbols[abs(L)]}{j}' for j in J])
        for ML in range(-L, L - 1, -1):
            for MS in np.arange(S, -S + 1, 1):
                if (ML, MS) in comb_counter:
                    comb_counter[ML, MS] -= 1
                    if comb_counter[ML, MS] == 0:
                        del comb_counter[ML, MS]
    return term_symbols