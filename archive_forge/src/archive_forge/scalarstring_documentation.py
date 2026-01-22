from __future__ import print_function, absolute_import, division, unicode_literals
from ruamel.yaml.compat import text_type
from ruamel.yaml.anchor import Anchor

    the routine here walks over a simple yaml tree (recursing in
    dict values and list items) and converts strings that
    have multiple lines to literal scalars

    You can also provide an explicit (ordered) mapping for multiple transforms
    (first of which is executed):
        map = ruamel.yaml.compat.ordereddict
        map['
'] = preserve_literal
        map[':'] = SingleQuotedScalarString
        walk_tree(data, map=map)
    