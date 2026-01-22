from nipype.utils.docparse import reverse_opt_map, build_doc, insert_doc
def test_insert_doc():
    new_items = ['infile : str', '    The name of the input file']
    new_items.extend(['outfile : str', '    The name of the output file'])
    newdoc = insert_doc(fmtd_doc, new_items)
    assert newdoc == inserted_doc