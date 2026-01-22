from ...cloudpath import CloudImplementation
from ..localclient import LocalClient
from ..localpath import LocalPath
Replacement for GSPath that uses the local file system. Intended as a monkeypatch substitute
    when writing tests.
    