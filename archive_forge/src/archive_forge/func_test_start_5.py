from prov.model import *
def test_start_5(self):
    document = self.new_document()
    document.start(EX_NS['a1'], trigger=EX_NS['e1'], identifier=EX_NS['start5'], starter=EX_NS['a2'])
    self.do_tests(document)