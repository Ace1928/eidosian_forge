from prov.model import *
def test_derivation_12(self):
    document = self.new_document()
    document.quotation(EX_NS['e2'], usedEntity=EX_NS['e1'], identifier=EX_NS['quo1'], activity=EX_NS['a'], usage=EX_NS['u'], generation=EX_NS['g'])
    self.do_tests(document)