from nipype.utils.docparse import reverse_opt_map, build_doc, insert_doc
def test_rev_opt_map():
    map = {'-f': 'fun', '-o': 'outline'}
    rev_map = reverse_opt_map(foo_opt_map)
    assert rev_map == map