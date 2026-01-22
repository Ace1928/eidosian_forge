from prov.model import *
def test_derivation_8(self):
    document = self.new_document()
    der = document.derivation(EX_NS['e2'], usedEntity=EX_NS['e1'], identifier=EX_NS['der8'])
    self.add_label(der)
    self.add_types(der)
    self.add_further_attributes(der)
    self.do_tests(document)