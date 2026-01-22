from prov.model import *
def test_communication_1(self):
    document = self.new_document()
    document.communication(EX_NS['a2'], None, identifier=EX_NS['inf1'])
    self.do_tests(document)