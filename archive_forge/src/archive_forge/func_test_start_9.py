from prov.model import *
def test_start_9(self):
    document = self.new_document()
    document.start(EX_NS['a1'], trigger=EX_NS['e1'])
    self.do_tests(document)