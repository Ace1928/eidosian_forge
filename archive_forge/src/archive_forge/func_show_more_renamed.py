from io import StringIO
from breezy import osutils, trace
from .bzr.inventorytree import InventoryTreeChange
def show_more_renamed(item):
    dec_new_path = decorate_path(item.path[1], item.kind[1], item.meta_modified())
    to_file.write(' => %s' % dec_new_path)
    if item.changed_content or item.meta_modified():
        extra_modified.append(InventoryTreeChange(item.file_id, (item.path[1], item.path[1]), item.changed_content, item.versioned, (item.parent_id[1], item.parent_id[1]), (item.name[1], item.name[1]), (item.kind[1], item.kind[1]), item.executable))