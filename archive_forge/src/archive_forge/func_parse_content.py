from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
import xml.etree.ElementTree as etree
import re
from typing import TYPE_CHECKING
def parse_content(self, parent: etree.Element, block: str) -> tuple[etree.Element | None, str, str]:
    """Get sibling admonition.

        Retrieve the appropriate sibling element. This can get tricky when
        dealing with lists.

        """
    old_block = block
    the_rest = ''
    if self.current_sibling is not None:
        sibling = self.current_sibling
        block, the_rest = self.detab(block, self.content_indent)
        self.current_sibling = None
        self.content_indent = 0
        return (sibling, block, the_rest)
    sibling = self.lastChild(parent)
    if sibling is None or sibling.tag != 'div' or sibling.get('class', '').find(self.CLASSNAME) == -1:
        sibling = None
    else:
        last_child = self.lastChild(sibling)
        indent = 0
        while last_child is not None:
            if sibling is not None and block.startswith(' ' * self.tab_length * 2) and (last_child is not None) and (last_child.tag in ('ul', 'ol', 'dl')):
                sibling = self.lastChild(last_child)
                last_child = self.lastChild(sibling) if sibling is not None else None
                block = block[self.tab_length:]
                indent += self.tab_length
            else:
                last_child = None
        if not block.startswith(' ' * self.tab_length):
            sibling = None
        if sibling is not None:
            indent += self.tab_length
            block, the_rest = self.detab(old_block, indent)
            self.current_sibling = sibling
            self.content_indent = indent
    return (sibling, block, the_rest)