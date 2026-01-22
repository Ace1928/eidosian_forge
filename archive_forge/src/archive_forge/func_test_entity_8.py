from prov.model import *
def test_entity_8(self):
    document = self.new_document()
    a = document.entity(EX_NS['e8'])
    a.add_attributes([(PROV_LABEL, 'entity8')])
    self.add_types(a)
    self.add_types(a)
    self.add_locations(a)
    self.add_locations(a)
    self.add_labels(a)
    self.add_labels(a)
    self.do_tests(document)