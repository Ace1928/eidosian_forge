from prov.model import *
def run_entity_with_one_type_attribute(self, n):
    document = self.new_document()
    document.entity(EX_NS['et%d' % n], {'prov:type': self.attribute_values[n]})
    self.do_tests(document)