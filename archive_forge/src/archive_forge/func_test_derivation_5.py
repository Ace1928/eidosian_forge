from prov.model import *
def test_derivation_5(self):
    document = self.new_document()
    document.derivation(EX_NS['e2'], usedEntity=EX_NS['e1'], identifier=EX_NS['der5'], activity=EX_NS['a'])
    self.do_tests(document)