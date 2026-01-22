from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
Parse the text bytes of a serialized inventory delta.

        If versioned_root and/or tree_references flags were set via
        require_flags, then the parsed flags must match or a BzrError will be
        raised.

        :param lines: The lines to parse. This can be obtained by calling
            delta_to_lines.
        :return: (parent_id, new_id, versioned_root, tree_references,
            inventory_delta)
        