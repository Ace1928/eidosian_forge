from prov.model import *
def test_association_7(self):
    document = self.new_document()
    assoc = document.association(EX_NS['a1'], agent=EX_NS['ag1'], identifier=EX_NS['assoc7'], plan=EX_NS['plan1'])
    self.add_labels(assoc)
    self.add_types(assoc)
    self.do_tests(document)