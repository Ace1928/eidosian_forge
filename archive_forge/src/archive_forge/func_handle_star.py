from __future__ import absolute_import
import re
import operator
import sys
def handle_star(next, token):
    """
    /*/
    """

    def select(result):
        for node in result:
            for name in node.child_attrs:
                for child in iterchildren(node, name):
                    yield child
    return select