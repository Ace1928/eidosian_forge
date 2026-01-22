from prov.model import *
def test_generation_5(self):
    document = self.new_document()
    a = document.generation(EX_NS['e1'], identifier=EX_NS['gen5'], activity=EX_NS['a1'], time=datetime.datetime.now())
    a.add_attributes([(PROV_ROLE, 'somerole')])
    self.add_types(a)
    self.add_locations(a)
    self.add_labels(a)
    self.add_further_attributes(a)
    self.do_tests(document)