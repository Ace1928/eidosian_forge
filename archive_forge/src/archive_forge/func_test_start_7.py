from prov.model import *
def test_start_7(self):
    document = self.new_document()
    document.start(EX_NS['a1'], identifier=EX_NS['start7'], starter=EX_NS['a2'], time=datetime.datetime.now())
    self.do_tests(document)