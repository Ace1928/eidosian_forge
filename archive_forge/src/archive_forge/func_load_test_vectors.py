import os
import re
import json
import errno
import binascii
import warnings
from binascii import unhexlify
from Cryptodome.Util.py3compat import FileNotFoundError
def load_test_vectors(dir_comps, file_name, description, conversions):
    """Load and parse a test vector file, formatted using the NIST style.

    Args:
        dir_comps (list of strings):
          The path components under the ``pycryptodome_test_vectors`` package.
          For instance ``("Cipher", "AES")``.
        file_name (string):
          The name of the file with the test vectors.
        description (string):
          A description applicable to the test vectors in the file.
        conversions (dictionary):
          The dictionary contains functions.
          Values in the file that have an entry in this dictionary
          will be converted usign the matching function.
          Otherwise, values will be considered as hexadecimal and
          converted to binary.

    Returns:
        A list of test vector objects.

    The file is formatted in the following way:

    - Lines starting with "#" are comments and will be ignored.
    - Each test vector is a sequence of 1 or more adjacent lines, where
      each lines is an assignement.
    - Test vectors are separated by an empty line, a comment, or
      a line starting with "[".

    A test vector object has the following attributes:

    - desc (string): description
    - counter (int): the order of the test vector in the file (from 1)
    - others (list): zero or more lines of the test vector that were not assignments
    - left-hand side of each assignment (lowercase): the value of the
      assignement, either converted or bytes.
    """
    results = None
    try:
        if not test_vectors_available:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)
        description = '%s test (%s)' % (description, file_name)
        init_dir = os.path.dirname(pycryptodome_test_vectors.__file__)
        full_file_name = os.path.join(os.path.join(init_dir, *dir_comps), file_name)
        with open(full_file_name) as file_in:
            results = _load_tests(dir_comps, file_in, description, conversions)
    except FileNotFoundError:
        warnings.warn('Warning: skipping extended tests for ' + description, UserWarning, stacklevel=2)
    return results