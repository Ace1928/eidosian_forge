import enum
import typing
@property
def nt_status(self) -> int:
    """The Windows NT Status code that represents this error."""
    codes = getattr(self, '_SSPI_CODE', self._error_code) or 4294967295
    return codes[0] if isinstance(codes, (list, tuple)) else codes