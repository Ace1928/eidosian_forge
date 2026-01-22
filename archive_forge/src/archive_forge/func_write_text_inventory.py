from .errors import BzrError
from .inventory import Inventory
def write_text_inventory(inv, outf):
    """Write out inv in a simple trad-unix text format."""
    outf.write(START_MARK)
    for path, ie in inv.iter_entries():
        if inv.is_root(ie.file_id):
            continue
        outf.write(ie.file_id + ' ')
        outf.write(escape(ie.name) + ' ')
        outf.write(ie.kind + ' ')
        outf.write(ie.parent_id + ' ')
        if ie.kind == 'file':
            outf.write(ie.text_id)
            outf.write(' ' + ie.text_sha1)
            outf.write(' ' + str(ie.text_size))
        outf.write('\n')
    outf.write(END_MARK)