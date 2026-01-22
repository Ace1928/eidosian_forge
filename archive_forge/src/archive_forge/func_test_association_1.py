from prov.model import *
def test_association_1(self):
    document = self.new_document()
    document.association(EX_NS['a1'], identifier=EX_NS['assoc1'])
    self.do_tests(document)