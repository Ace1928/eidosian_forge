import math
import warnings
from Bio import SeqUtils, Seq
from Bio import BiopythonWarning
def salt_correction(Na=0, K=0, Tris=0, Mg=0, dNTPs=0, method=1, seq=None):
    """Calculate a term to correct Tm for salt ions.

    Depending on the Tm calculation, the term will correct Tm or entropy. To
    calculate corrected Tm values, different operations need to be applied:

     - methods 1-4: Tm(new) = Tm(old) + corr
     - method 5: deltaS(new) = deltaS(old) + corr
     - methods 6+7: Tm(new) = 1/(1/Tm(old) + corr)

    Arguments:
     - Na, K, Tris, Mg, dNTPS: Millimolar concentration of respective ion. To
       have a simple 'salt correction', just pass Na. If any of K, Tris, Mg and
       dNTPS is non-zero, a 'sodium-equivalent' concentration is calculated
       according to von Ahsen et al. (2001, Clin Chem 47: 1956-1961):
       [Na_eq] = [Na+] + [K+] + [Tris]/2 + 120*([Mg2+] - [dNTPs])^0.5
       If [dNTPs] >= [Mg2+]: [Na_eq] = [Na+] + [K+] + [Tris]/2
     - method: Which method to be applied. Methods 1-4 correct Tm, method 5
       corrects deltaS, methods 6 and 7 correct 1/Tm. The methods are:

       1. 16.6 x log[Na+]
          (Schildkraut & Lifson (1965), Biopolymers 3: 195-208)
       2. 16.6 x log([Na+]/(1.0 + 0.7*[Na+]))
          (Wetmur (1991), Crit Rev Biochem Mol Biol 126: 227-259)
       3. 12.5 x log(Na+]
          (SantaLucia et al. (1996), Biochemistry 35: 3555-3562
       4. 11.7 x log[Na+]
          (SantaLucia (1998), Proc Natl Acad Sci USA 95: 1460-1465
       5. Correction for deltaS: 0.368 x (N-1) x ln[Na+]
          (SantaLucia (1998), Proc Natl Acad Sci USA 95: 1460-1465)
       6. (4.29(%GC)-3.95)x1e-5 x ln[Na+] + 9.40e-6 x ln[Na+]^2
          (Owczarzy et al. (2004), Biochemistry 43: 3537-3554)
       7. Complex formula with decision tree and 7 empirical constants.
          Mg2+ is corrected for dNTPs binding (if present)
          (Owczarzy et al. (2008), Biochemistry 47: 5336-5353)

    Examples
    --------
    >>> from Bio.SeqUtils.MeltingTemp import salt_correction
    >>> print('%0.2f' % salt_correction(Na=50, method=1))
    -21.60
    >>> print('%0.2f' % salt_correction(Na=50, method=2))
    -21.85
    >>> print('%0.2f' % salt_correction(Na=100, Tris=20, method=2))
    -16.45
    >>> print('%0.2f' % salt_correction(Na=100, Tris=20, Mg=1.5, method=2))
    -10.99

    """
    if method in (5, 6, 7) and (not seq):
        raise ValueError('sequence is missing (is needed to calculate GC content or sequence length).')
    corr = 0
    if not method:
        return corr
    Mon = Na + K + Tris / 2.0
    mg = Mg * 0.001
    if sum((K, Mg, Tris, dNTPs)) > 0 and method != 7 and (dNTPs < Mg):
        Mon += 120 * math.sqrt(Mg - dNTPs)
    mon = Mon * 0.001
    if method in range(1, 7) and (not mon):
        raise ValueError('Total ion concentration of zero is not allowed in this method.')
    if method == 1:
        corr = 16.6 * math.log10(mon)
    if method == 2:
        corr = 16.6 * math.log10(mon / (1.0 + 0.7 * mon))
    if method == 3:
        corr = 12.5 * math.log10(mon)
    if method == 4:
        corr = 11.7 * math.log10(mon)
    if method == 5:
        corr = 0.368 * (len(seq) - 1) * math.log(mon)
    if method == 6:
        corr = (4.29 * SeqUtils.gc_fraction(seq, 'ignore') - 3.95) * 1e-05 * math.log(mon) + 9.4e-06 * math.log(mon) ** 2
    if method == 7:
        a, b, c, d = (3.92, -0.911, 6.26, 1.42)
        e, f, g = (-48.2, 52.5, 8.31)
        if dNTPs > 0:
            dntps = dNTPs * 0.001
            ka = 30000.0
            mg = (-(ka * dntps - ka * mg + 1.0) + math.sqrt((ka * dntps - ka * mg + 1.0) ** 2 + 4.0 * ka * mg)) / (2.0 * ka)
        if Mon > 0:
            R = math.sqrt(mg) / mon
            if R < 0.22:
                corr = (4.29 * SeqUtils.gc_fraction(seq, 'ignore') - 3.95) * 1e-05 * math.log(mon) + 9.4e-06 * math.log(mon) ** 2
                return corr
            elif R < 6.0:
                a = 3.92 * (0.843 - 0.352 * math.sqrt(mon) * math.log(mon))
                d = 1.42 * (1.279 - 0.00403 * math.log(mon) - 0.00803 * math.log(mon) ** 2)
                g = 8.31 * (0.486 - 0.258 * math.log(mon) + 0.00525 * math.log(mon) ** 3)
        corr = (a + b * math.log(mg) + SeqUtils.gc_fraction(seq, 'ignore') * (c + d * math.log(mg)) + 1 / (2.0 * (len(seq) - 1)) * (e + f * math.log(mg) + g * math.log(mg) ** 2)) * 1e-05
    if method > 7:
        raise ValueError("Allowed values for parameter 'method' are 1-7.")
    return corr