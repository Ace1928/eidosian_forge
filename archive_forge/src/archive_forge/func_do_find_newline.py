import collections
import locale
import sys
from typing import Dict, List
from charset_normalizer import from_path  # pylint: disable=import-error
def do_find_newline(self, source: List[str]) -> str:
    """Return type of newline used in source.

        Parameters
        ----------
        source : list
            A list of lines.

        Returns
        -------
        newline : str
            The most prevalent new line type found.
        """
    assert not isinstance(source, unicode)
    counter: Dict[str, int] = collections.defaultdict(int)
    for line in source:
        if line.endswith(self.CRLF):
            counter[self.CRLF] += 1
        elif line.endswith(self.CR):
            counter[self.CR] += 1
        elif line.endswith(self.LF):
            counter[self.LF] += 1
    return (sorted(counter, key=counter.get, reverse=True) or [self.LF])[0]