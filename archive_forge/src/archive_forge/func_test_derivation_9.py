from prov.model import *
def test_derivation_9(self):
    document = self.new_document()
    der = document.derivation(EX_NS['e2'], usedEntity=None)
    self.add_types(der)
    self.do_tests(document)