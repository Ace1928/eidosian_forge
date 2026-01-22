from struct import Struct
Return byte string of length `n_bytes` at current position

        Returns sub-string from ``self.buf`` and updates ``self.ptr`` to the
        position after the read data.

        Parameters
        ----------
        n_bytes : int, optional
           number of bytes to read.  Can be -1 (the default) in which
           case we return all the remaining bytes in ``self.buf``

        Returns
        -------
        s : byte string
        