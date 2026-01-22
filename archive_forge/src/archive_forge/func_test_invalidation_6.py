from prov.model import *
def test_invalidation_6(self):
    document = self.new_document()
    document.invalidation(EX_NS['e1'], activity=EX_NS['a1'])
    self.do_tests(document)