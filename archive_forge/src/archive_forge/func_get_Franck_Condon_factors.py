from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
def get_Franck_Condon_factors(self, temperature, forces, order=1):
    """Return FC factors and corresponding frequencies up to given order.

        Parameters
        ----------
        temperature: float
          Temperature in K. Vibronic levels are occupied by a
          Boltzman distribution.
        forces: array
          Forces on atoms in the exited electronic state
        order: int
          number of quanta taken into account, default

        Returns
        --------
        FC: 3 entry list
          FC[0] = FC factors for 0-0 and +-1 vibrational quantum
          FC[1] = FC factors for +-2 vibrational quanta
          FC[2] = FC factors for combinations
        frequencies: 3 entry list
          frequencies[0] correspond to FC[0]
          frequencies[1] correspond to FC[1]
          frequencies[2] correspond to FC[2]
        """
    S, f = self.get_Huang_Rhys_factors(forces)
    assert order > 0
    n = order + 1
    T = temperature
    freq = np.array(f)
    freq_n = [[] * i for i in range(n - 1)]
    freq_neg = [[] * i for i in range(n - 1)]
    for i in range(1, n):
        freq_n[i - 1] = freq * i
        freq_neg[i - 1] = freq * -i
    freq_nn = [x for x in combinations(chain(*freq_n), 2)]
    for i in range(len(freq_nn)):
        freq_nn[i] = freq_nn[i][0] + freq_nn[i][1]
    indices2 = []
    for i, y in enumerate(freq):
        ind = [j for j, x in enumerate(freq_nn) if y == 0 or x % y == 0]
        indices2.append(ind)
    indices2 = [x for x in chain(*indices2)]
    freq_nn = np.delete(freq_nn, indices2)
    frequencies = [[] * x for x in range(3)]
    frequencies[0].append(freq_neg[0])
    frequencies[0].append([0])
    frequencies[0].append(freq_n[0])
    frequencies[0] = [x for x in chain(*frequencies[0])]
    for i in range(1, n - 1):
        frequencies[1].append(freq_neg[i])
        frequencies[1].append(freq_n[i])
    frequencies[1] = [x for x in chain(*frequencies[1])]
    frequencies[2] = freq_nn
    E = freq / 8065.5
    f_n = [[] * i for i in range(n)]
    for j in range(0, n):
        f_n[j] = np.exp(-E * j / (kB * T))
    Z = np.empty(len(S))
    Z = np.sum(f_n, 0)
    w_n = [[] * k for k in range(n)]
    for l in range(n):
        w_n[l] = f_n[l] / Z
    O_n = [[] * m for m in range(n)]
    O_neg = [[] * m for m in range(n)]
    for o in range(n):
        O_n[o] = [[] * p for p in range(n)]
        O_neg[o] = [[] * p for p in range(n - 1)]
        for q in range(o, n + o):
            a = np.minimum(o, q)
            summe = []
            for k in range(a + 1):
                s = (-1) ** (q - k) * np.sqrt(S) ** (o + q - 2 * k) * factorial(o) * factorial(q) / (factorial(k) * factorial(o - k) * factorial(q - k))
                summe.append(s)
            summe = np.sum(summe, 0)
            O_n[o][q - o] = (np.exp(-S / 2) / (factorial(o) * factorial(q)) ** 0.5 * summe) ** 2 * w_n[o]
        for q in range(n - 1):
            O_neg[o][q] = [0 * b for b in range(len(S))]
        for q in range(o - 1, -1, -1):
            a = np.minimum(o, q)
            summe = []
            for k in range(a + 1):
                s = (-1) ** (q - k) * np.sqrt(S) ** (o + q - 2 * k) * factorial(o) * factorial(q) / (factorial(k) * factorial(o - k) * factorial(q - k))
                summe.append(s)
            summe = np.sum(summe, 0)
            O_neg[o][q] = (np.exp(-S / 2) / (factorial(o) * factorial(q)) ** 0.5 * summe) ** 2 * w_n[o]
    O_neg = np.delete(O_neg, 0, 0)
    FC_n = [[] * i for i in range(n)]
    FC_n = np.sum(O_n, 0)
    zero = reduce(mul, FC_n[0])
    FC_neg = [[] * i for i in range(n - 2)]
    FC_neg = np.sum(O_neg, 0)
    FC_n = np.delete(FC_n, 0, 0)
    FC_nn = [x for x in combinations(chain(*FC_n), 2)]
    for i in range(len(FC_nn)):
        FC_nn[i] = FC_nn[i][0] * FC_nn[i][1]
    FC_nn = np.delete(FC_nn, indices2)
    FC = [[] * x for x in range(3)]
    FC[0].append(FC_neg[0])
    FC[0].append([zero])
    FC[0].append(FC_n[0])
    FC[0] = [x for x in chain(*FC[0])]
    for i in range(1, n - 1):
        FC[1].append(FC_neg[i])
        FC[1].append(FC_n[i])
    FC[1] = [x for x in chain(*FC[1])]
    FC[2] = FC_nn
    return (FC, frequencies)