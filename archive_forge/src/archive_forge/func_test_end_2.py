from prov.model import *
def test_end_2(self):
    document = self.new_document()
    document.end(EX_NS['a1'], trigger=EX_NS['e1'], identifier=EX_NS['end2'])
    self.do_tests(document)