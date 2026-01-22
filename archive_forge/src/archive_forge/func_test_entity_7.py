from prov.model import *
def test_entity_7(self):
    document = self.new_document()
    a = document.entity(EX_NS['e7'])
    a.add_attributes([(PROV_LABEL, 'entity7')])
    self.add_types(a)
    self.add_locations(a)
    self.add_labels(a)
    self.do_tests(document)