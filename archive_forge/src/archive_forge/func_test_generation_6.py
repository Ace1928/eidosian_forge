from prov.model import *
def test_generation_6(self):
    document = self.new_document()
    document.generation(EX_NS['e1'], activity=EX_NS['a1'], time=datetime.datetime.now())
    self.do_tests(document)