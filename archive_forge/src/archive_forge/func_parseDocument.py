from __future__ import annotations
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Iterable, Any
from . import util
def parseDocument(self, lines: Iterable[str]) -> etree.ElementTree:
    """ Parse a Markdown document into an `ElementTree`.

        Given a list of lines, an `ElementTree` object (not just a parent
        `Element`) is created and the root element is passed to the parser
        as the parent. The `ElementTree` object is returned.

        This should only be called on an entire document, not pieces.

        Arguments:
            lines: A list of lines (strings).

        Returns:
            An element tree.
        """
    self.root = etree.Element(self.md.doc_tag)
    self.parseChunk(self.root, '\n'.join(lines))
    return etree.ElementTree(self.root)