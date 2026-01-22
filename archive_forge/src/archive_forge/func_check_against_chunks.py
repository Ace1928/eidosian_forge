import hashlib
from typing import TYPE_CHECKING, BinaryIO, Dict, Iterable, List, Optional
from pip._internal.exceptions import HashMismatch, HashMissing, InstallationError
from pip._internal.utils.misc import read_chunks
def check_against_chunks(self, chunks: Iterable[bytes]) -> None:
    """Check good hashes against ones built from iterable of chunks of
        data.

        Raise HashMismatch if none match.

        """
    gots = {}
    for hash_name in self._allowed.keys():
        try:
            gots[hash_name] = hashlib.new(hash_name)
        except (ValueError, TypeError):
            raise InstallationError(f'Unknown hash name: {hash_name}')
    for chunk in chunks:
        for hash in gots.values():
            hash.update(chunk)
    for hash_name, got in gots.items():
        if got.hexdigest() in self._allowed[hash_name]:
            return
    self._raise(gots)