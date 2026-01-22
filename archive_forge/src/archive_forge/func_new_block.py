import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def new_block(self, package=None, version=None, distributions=None, urgency=None, urgency_comment=None, changes=None, author=None, date=None, other_pairs=None, encoding=None):
    """ Add a new changelog block to the changelog

        Start a new :class:`ChangeBlock` entry representing a new version
        of the package. The arguments (all optional) are passed directly
        to the :class:`ChangeBlock` constructor; they specify the values
        that can be provided to the `set_*` methods of this class. If
        they are omitted the associated attributes *must* be assigned to
        before the changelog is formatted as a str or written to a file.
        """
    encoding = encoding or self._encoding
    block = ChangeBlock(package, version, distributions, urgency, urgency_comment, changes, author, date, other_pairs, encoding)
    if self._blocks:
        block.add_trailing_line('')
    self._blocks.insert(0, block)