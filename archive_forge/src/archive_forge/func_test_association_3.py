from prov.model import *
def test_association_3(self):
    document = self.new_document()
    document.association(EX_NS['a1'], agent=EX_NS['ag1'], identifier=EX_NS['assoc3'])
    self.do_tests(document)