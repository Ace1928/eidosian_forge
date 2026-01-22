from prov.model import *
def test_derivation_3(self):
    document = self.new_document()
    document.derivation(EX_NS['e2'], usedEntity=EX_NS['e1'], identifier=EX_NS['der3'])
    self.do_tests(document)