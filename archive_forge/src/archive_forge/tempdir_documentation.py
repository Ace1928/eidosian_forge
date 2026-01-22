import os as _os
from pathlib import Path
from tempfile import TemporaryDirectory

        Open a file named `filename` in a temporary directory.

        This context manager is preferred over `NamedTemporaryFile` in
        stdlib `tempfile` when one needs to reopen the file.

        Arguments `mode` and `bufsize` are passed to `open`.
        Rest of the arguments are passed to `TemporaryDirectory`.

        