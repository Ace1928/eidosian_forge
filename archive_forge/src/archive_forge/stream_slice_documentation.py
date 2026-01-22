from apitools.base.py import exceptions
Read at most size bytes from this slice.

        Compared to other streams, there is one case where we may
        unexpectedly raise an exception on read: if the underlying stream
        is exhausted (i.e. returns no bytes on read), and the size of this
        slice indicates we should still be able to read more bytes, we
        raise exceptions.StreamExhausted.

        Args:
          size: If provided, read no more than size bytes from the stream.

        Returns:
          The bytes read from this slice.

        Raises:
          exceptions.StreamExhausted

        