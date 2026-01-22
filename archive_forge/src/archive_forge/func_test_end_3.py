from prov.model import *
def test_end_3(self):
    document = self.new_document()
    document.end(EX_NS['a1'], identifier=EX_NS['end3'])
    self.do_tests(document)