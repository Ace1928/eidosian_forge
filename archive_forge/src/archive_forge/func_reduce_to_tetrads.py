import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
def reduce_to_tetrads(chords, keep_bass=False):
    """
    Reduce chords to tetrads.

    The function follows the reduction rules implemented in [1]_. If a chord
    does not contain a third, major second or fourth, it is reduced to a power
    chord. If it does not contain neither a third nor a fifth, it is reduced
    to a single note "chord".

    Parameters
    ----------
    chords : numpy structured array
        Chords to be reduced.
    keep_bass : bool
        Indicates whether to keep the bass note or set it to 0.

    Returns
    -------
    reduced_chords : numpy structured array
        Chords reduced to tetrads.

    References
    ----------
    .. [1] Johan Pauwels and Geoffroy Peeters.
           "Evaluating Automatically Estimated Chord Sequences."
           In Proceedings of ICASSP 2013, Vancouver, Canada, 2013.

    """
    unison = chords['intervals'][:, 0].astype(bool)
    maj_sec = chords['intervals'][:, 2].astype(bool)
    min_third = chords['intervals'][:, 3].astype(bool)
    maj_third = chords['intervals'][:, 4].astype(bool)
    perf_fourth = chords['intervals'][:, 5].astype(bool)
    dim_fifth = chords['intervals'][:, 6].astype(bool)
    perf_fifth = chords['intervals'][:, 7].astype(bool)
    aug_fifth = chords['intervals'][:, 8].astype(bool)
    maj_sixth = chords['intervals'][:, 9].astype(bool)
    dim_seventh = maj_sixth
    min_seventh = chords['intervals'][:, 10].astype(bool)
    maj_seventh = chords['intervals'][:, 11].astype(bool)
    no_chord = (chords['intervals'] == NO_CHORD[-1]).all(axis=1)
    reduced_chords = chords.copy()
    ivs = reduced_chords['intervals']
    ivs[~no_chord] = interval_list('(1)')
    ivs[unison & perf_fifth] = interval_list('(1,5)')
    sus2 = ~perf_fourth & maj_sec
    sus2_ivs = _shorthands['sus2']
    ivs[sus2] = sus2_ivs
    ivs[sus2 & maj_sixth] = interval_list('(6)', sus2_ivs.copy())
    ivs[sus2 & maj_seventh] = interval_list('(7)', sus2_ivs.copy())
    ivs[sus2 & min_seventh] = interval_list('(b7)', sus2_ivs.copy())
    sus4 = perf_fourth & ~maj_sec
    sus4_ivs = _shorthands['sus4']
    ivs[sus4] = sus4_ivs
    ivs[sus4 & maj_sixth] = interval_list('(6)', sus4_ivs.copy())
    ivs[sus4 & maj_seventh] = interval_list('(7)', sus4_ivs.copy())
    ivs[sus4 & min_seventh] = interval_list('(b7)', sus4_ivs.copy())
    ivs[min_third] = _shorthands['min']
    ivs[min_third & maj_sixth] = _shorthands['min6']
    ivs[min_third & maj_seventh] = _shorthands['minmaj7']
    ivs[min_third & min_seventh] = _shorthands['min7']
    minaugfifth = min_third & ~perf_fifth & aug_fifth
    ivs[minaugfifth] = interval_list('(1,b3,#5)')
    ivs[minaugfifth & maj_seventh] = interval_list('(1,b3,#5,7)')
    ivs[minaugfifth & min_seventh] = interval_list('(1,b3,#5,b7)')
    mindimfifth = min_third & ~perf_fifth & dim_fifth
    ivs[mindimfifth] = _shorthands['dim']
    ivs[mindimfifth & dim_seventh] = _shorthands['dim7']
    ivs[mindimfifth & min_seventh] = _shorthands['hdim7']
    ivs[maj_third] = _shorthands['maj']
    ivs[maj_third & maj_sixth] = _shorthands['maj6']
    ivs[maj_third & maj_seventh] = _shorthands['maj7']
    ivs[maj_third & min_seventh] = _shorthands['7']
    majdimfifth = maj_third & ~perf_fifth & dim_fifth
    ivs[majdimfifth] = interval_list('(1,3,b5)')
    ivs[majdimfifth & maj_seventh] = interval_list('(1,3,b5,7)')
    ivs[majdimfifth & min_seventh] = interval_list('(1,3,b5,b7)')
    majaugfifth = maj_third & ~perf_fifth & aug_fifth
    aug_ivs = _shorthands['aug']
    ivs[majaugfifth] = _shorthands['aug']
    ivs[majaugfifth & maj_seventh] = interval_list('(7)', aug_ivs.copy())
    ivs[majaugfifth & min_seventh] = interval_list('(b7)', aug_ivs.copy())
    if not keep_bass:
        reduced_chords['bass'] = 0
    else:
        reduced_chords['bass'] *= ivs[range(len(reduced_chords)), reduced_chords['bass']]
    reduced_chords['bass'][no_chord] = -1
    return reduced_chords