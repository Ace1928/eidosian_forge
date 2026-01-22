import re
from qiskit.result import postprocess
from qiskit import exceptions
def int_outcomes(self):
    """Build a counts dictionary with integer keys instead of count strings

        Returns:
            dict: A dictionary with the keys as integers instead of bitstrings
        Raises:
            QiskitError: If the Counts object contains counts for dit strings
        """
    if self.int_raw:
        return self.int_raw
    else:
        out_dict = {}
        for bitstring, value in self.items():
            if not self.bitstring_regex.search(bitstring):
                raise exceptions.QiskitError('Counts objects with dit strings do not currently support conversion to integer')
            int_key = self._remove_space_underscore(bitstring)
            out_dict[int_key] = value
        return out_dict