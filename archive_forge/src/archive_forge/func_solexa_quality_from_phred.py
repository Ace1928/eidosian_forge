import warnings
from math import log
from Bio import BiopythonParserWarning
from Bio import BiopythonWarning
from Bio import StreamModeError
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from typing import (
def solexa_quality_from_phred(phred_quality: float) -> float:
    """Convert a PHRED quality (range 0 to about 90) to a Solexa quality.

    PHRED and Solexa quality scores are both log transformations of a
    probality of error (high score = low probability of error). This function
    takes a PHRED score, transforms it back to a probability of error, and
    then re-expresses it as a Solexa score. This assumes the error estimates
    are equivalent.

    How does this work exactly? Well the PHRED quality is minus ten times the
    base ten logarithm of the probability of error::

        phred_quality = -10*log(error,10)

    Therefore, turning this round::

        error = 10 ** (- phred_quality / 10)

    Now, Solexa qualities use a different log transformation::

        solexa_quality = -10*log(error/(1-error),10)

    After substitution and a little manipulation we get::

         solexa_quality = 10*log(10**(phred_quality/10.0) - 1, 10)

    However, real Solexa files use a minimum quality of -5. This does have a
    good reason - a random base call would be correct 25% of the time,
    and thus have a probability of error of 0.75, which gives 1.25 as the PHRED
    quality, or -4.77 as the Solexa quality. Thus (after rounding), a random
    nucleotide read would have a PHRED quality of 1, or a Solexa quality of -5.

    Taken literally, this logarithic formula would map a PHRED quality of zero
    to a Solexa quality of minus infinity. Of course, taken literally, a PHRED
    score of zero means a probability of error of one (i.e. the base call is
    definitely wrong), which is worse than random! In practice, a PHRED quality
    of zero usually means a default value, or perhaps random - and therefore
    mapping it to the minimum Solexa score of -5 is reasonable.

    In conclusion, we follow EMBOSS, and take this logarithmic formula but also
    apply a minimum value of -5.0 for the Solexa quality, and also map a PHRED
    quality of zero to -5.0 as well.

    Note this function will return a floating point number, it is up to you to
    round this to the nearest integer if appropriate.  e.g.

    >>> print("%0.2f" % round(solexa_quality_from_phred(80), 2))
    80.00
    >>> print("%0.2f" % round(solexa_quality_from_phred(50), 2))
    50.00
    >>> print("%0.2f" % round(solexa_quality_from_phred(20), 2))
    19.96
    >>> print("%0.2f" % round(solexa_quality_from_phred(10), 2))
    9.54
    >>> print("%0.2f" % round(solexa_quality_from_phred(5), 2))
    3.35
    >>> print("%0.2f" % round(solexa_quality_from_phred(4), 2))
    1.80
    >>> print("%0.2f" % round(solexa_quality_from_phred(3), 2))
    -0.02
    >>> print("%0.2f" % round(solexa_quality_from_phred(2), 2))
    -2.33
    >>> print("%0.2f" % round(solexa_quality_from_phred(1), 2))
    -5.00
    >>> print("%0.2f" % round(solexa_quality_from_phred(0), 2))
    -5.00

    Notice that for high quality reads PHRED and Solexa scores are numerically
    equal. The differences are important for poor quality reads, where PHRED
    has a minimum of zero but Solexa scores can be negative.

    Finally, as a special case where None is used for a "missing value", None
    is returned:

    >>> print(solexa_quality_from_phred(None))
    None
    """
    if phred_quality is None:
        return None
    elif phred_quality > 0:
        return max(-5.0, 10 * log(10 ** (phred_quality / 10.0) - 1, 10))
    elif phred_quality == 0:
        return -5.0
    else:
        raise ValueError(f'PHRED qualities must be positive (or zero), not {phred_quality!r}')