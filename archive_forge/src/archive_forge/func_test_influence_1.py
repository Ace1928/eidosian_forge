from prov.model import *
def test_influence_1(self):
    document = self.new_document()
    document.influence(EX_NS['a2'], None, identifier=EX_NS['inf1'])
    self.do_tests(document)