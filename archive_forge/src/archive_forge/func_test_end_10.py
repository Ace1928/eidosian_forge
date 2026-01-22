from prov.model import *
def test_end_10(self):
    document = self.new_document()
    end = document.end(EX_NS['a1'], ender=EX_NS['a2'], time=datetime.datetime.now())
    end.add_attributes([(PROV_ROLE, 'yolk'), (PROV_ROLE, 'white')])
    self.add_types(end)
    self.add_locations(end)
    self.add_labels(end)
    self.add_further_attributes(end)
    self.do_tests(document)