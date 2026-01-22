from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def delta_to_lines(self, old_name, new_name, delta_to_new):
    """Return a line sequence for delta_to_new.

        Both the versioned_root and tree_references flags must be set via
        require_flags before calling this.

        :param old_name: A UTF8 revision id for the old inventory.  May be
            NULL_REVISION if there is no older inventory and delta_to_new
            includes the entire inventory contents.
        :param new_name: The version name of the inventory we create with this
            delta.
        :param delta_to_new: An inventory delta such as Inventory.apply_delta
            takes.
        :return: The serialized delta as lines.
        """
    if not isinstance(old_name, bytes):
        raise TypeError('old_name should be str, got {!r}'.format(old_name))
    if not isinstance(new_name, bytes):
        raise TypeError('new_name should be str, got {!r}'.format(new_name))
    lines = [b'', b'', b'', b'', b'']
    to_line = self._delta_item_to_line
    for delta_item in delta_to_new:
        line = to_line(delta_item, new_name)
        if line.__class__ != bytes:
            raise InventoryDeltaError('to_line gave non-bytes output %(line)r', line=lines[-1])
        lines.append(line)
    lines.sort()
    lines[0] = b'format: %s\n' % FORMAT_1
    lines[1] = b'parent: %s\n' % old_name
    lines[2] = b'version: %s\n' % new_name
    lines[3] = b'versioned_root: %s\n' % self._serialize_bool(self._versioned_root)
    lines[4] = b'tree_references: %s\n' % self._serialize_bool(self._tree_references)
    return lines