import numpy
def writeRecord(self, s):
    """Write a record with the given bytes.

        Parameters
        ----------
        s : the string to write

        """
    length_bytes = len(s)
    self._write_check(length_bytes)
    self.write(s)
    self._write_check(length_bytes)