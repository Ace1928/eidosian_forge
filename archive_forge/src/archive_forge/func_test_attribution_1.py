from prov.model import *
def test_attribution_1(self):
    document = self.new_document()
    document.attribution(EX_NS['e1'], None, identifier=EX_NS['attr1'])
    self.do_tests(document)