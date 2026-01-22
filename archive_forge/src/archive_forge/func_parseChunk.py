from __future__ import annotations
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Iterable, Any
from . import util
def parseChunk(self, parent: etree.Element, text: str) -> None:
    """ Parse a chunk of Markdown text and attach to given `etree` node.

        While the `text` argument is generally assumed to contain multiple
        blocks which will be split on blank lines, it could contain only one
        block. Generally, this method would be called by extensions when
        block parsing is required.

        The `parent` `etree` Element passed in is altered in place.
        Nothing is returned.

        Arguments:
            parent: The parent element.
            text: The text to parse.

        """
    self.parseBlocks(parent, text.split('\n\n'))