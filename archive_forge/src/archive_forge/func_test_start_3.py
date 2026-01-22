from prov.model import *
def test_start_3(self):
    document = self.new_document()
    document.start(EX_NS['a1'], identifier=EX_NS['start3'])
    self.do_tests(document)