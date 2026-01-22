from prov.model import *
def test_end_6(self):
    document = self.new_document()
    document.end(EX_NS['a1'], identifier=EX_NS['end6'], ender=EX_NS['a2'])
    self.do_tests(document)