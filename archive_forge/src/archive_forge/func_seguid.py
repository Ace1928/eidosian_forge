import binascii
def seguid(seq):
    """Return the SEGUID (string) for a sequence (string or Seq object).

    Given a nucleotide or amino-acid sequence (or any string),
    returns the SEGUID string (A SEquence Globally Unique IDentifier).
    seq type = str.

    Note that the case is not important:

    >>> seguid("ACGTACGTACGT")
    'If6HIvcnRSQDVNiAoefAzySc6i4'
    >>> seguid("acgtACGTacgt")
    'If6HIvcnRSQDVNiAoefAzySc6i4'

    For more information about SEGUID, see:
    http://bioinformatics.anl.gov/seguid/
    https://doi.org/10.1002/pmic.200600032
    """
    import hashlib
    import base64
    m = hashlib.sha1()
    try:
        seq = bytes(seq)
    except TypeError:
        seq = seq.encode()
    m.update(seq.upper())
    tmp = base64.encodebytes(m.digest())
    return tmp.decode().replace('\n', '').rstrip('=')