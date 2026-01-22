from prov.model import *
def test_association_5(self):
    document = self.new_document()
    document.association(EX_NS['a1'], agent=EX_NS['ag1'])
    self.do_tests(document)