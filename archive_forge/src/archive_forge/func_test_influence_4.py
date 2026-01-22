from prov.model import *
def test_influence_4(self):
    document = self.new_document()
    document.influence(EX_NS['a2'], EX_NS['a1'])
    self.do_tests(document)