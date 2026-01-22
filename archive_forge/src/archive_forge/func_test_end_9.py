from prov.model import *
def test_end_9(self):
    document = self.new_document()
    document.end(EX_NS['a1'], trigger=EX_NS['e1'])
    self.do_tests(document)