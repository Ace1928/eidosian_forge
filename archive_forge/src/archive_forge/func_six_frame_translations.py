import re
import warnings
from math import pi, sin, cos, log, exp
from Bio.Seq import Seq, complement, complement_rna, translate
from Bio.Data import IUPACData
from Bio.Data.CodonTable import standard_dna_table
from Bio import BiopythonDeprecationWarning
def six_frame_translations(seq, genetic_code=1):
    """Return pretty string showing the 6 frame translations and GC content.

    Nice looking 6 frame translation with GC content - code from xbbtools
    similar to DNA Striders six-frame translation

    >>> from Bio.SeqUtils import six_frame_translations
    >>> print(six_frame_translations("AUGGCCAUUGUAAUGGGCCGCUGA"))
    GC_Frame: a:5 t:0 g:8 c:5
    Sequence: auggccauug ... gggccgcuga, 24 nt, 54.17 %GC
    <BLANKLINE>
    <BLANKLINE>
    1/1
      G  H  C  N  G  P  L
     W  P  L  *  W  A  A
    M  A  I  V  M  G  R  *
    auggccauuguaaugggccgcuga   54 %
    uaccgguaacauuacccggcgacu
    A  M  T  I  P  R  Q
     H  G  N  Y  H  A  A  S
      P  W  Q  L  P  G  S
    <BLANKLINE>
    <BLANKLINE>

    """
    from Bio.Seq import reverse_complement, reverse_complement_rna, translate
    if 'u' in seq.lower():
        anti = reverse_complement_rna(seq)
    else:
        anti = reverse_complement(seq)
    comp = anti[::-1]
    length = len(seq)
    frames = {}
    for i in range(3):
        fragment_length = 3 * ((length - i) // 3)
        frames[i + 1] = translate(seq[i:i + fragment_length], genetic_code)
        frames[-(i + 1)] = translate(anti[i:i + fragment_length], genetic_code)[::-1]
    if length > 20:
        short = f'{seq[:10]} ... {seq[-10:]}'
    else:
        short = seq
    header = 'GC_Frame:'
    for nt in ['a', 't', 'g', 'c']:
        header += ' %s:%d' % (nt, seq.count(nt.upper()))
    gc = 100 * gc_fraction(seq, ambiguous='ignore')
    header += '\nSequence: %s, %d nt, %0.2f %%GC\n\n\n' % (short.lower(), length, gc)
    res = header
    for i in range(0, length, 60):
        subseq = seq[i:i + 60]
        csubseq = comp[i:i + 60]
        p = i // 3
        res += '%d/%d\n' % (i + 1, i / 3 + 1)
        res += '  ' + '  '.join(frames[3][p:p + 20]) + '\n'
        res += ' ' + '  '.join(frames[2][p:p + 20]) + '\n'
        res += '  '.join(frames[1][p:p + 20]) + '\n'
        res += subseq.lower() + '%5d %%\n' % int(gc)
        res += csubseq.lower() + '\n'
        res += '  '.join(frames[-2][p:p + 20]) + '\n'
        res += ' ' + '  '.join(frames[-1][p:p + 20]) + '\n'
        res += '  ' + '  '.join(frames[-3][p:p + 20]) + '\n\n'
    return res