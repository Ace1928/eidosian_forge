import re
from qiskit.result import postprocess
from qiskit import exceptions
def most_frequent(self):
    """Return the most frequent count

        Returns:
            str: The bit string for the most frequent result
        Raises:
            QiskitError: when there is >1 count with the same max counts, or
                an empty object.
        """
    if not self:
        raise exceptions.QiskitError('Can not return a most frequent count on an empty object')
    max_value = max(self.values())
    max_values_counts = [x[0] for x in self.items() if x[1] == max_value]
    if len(max_values_counts) != 1:
        raise exceptions.QiskitError('Multiple values have the same maximum counts: %s' % ','.join(max_values_counts))
    return max_values_counts[0]