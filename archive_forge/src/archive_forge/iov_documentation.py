import enum
import typing
Results of an IOV operation.

    Unlike :class:`IOVBuffer` this limits the value of `data` to just an
    optionally set bytes. It is used as the return value of an IOV operation to
    better match what the expected values would be.
    