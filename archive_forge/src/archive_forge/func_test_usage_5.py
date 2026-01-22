from prov.model import *
def test_usage_5(self):
    document = self.new_document()
    use = document.usage(EX_NS['a1'], entity=EX_NS['e1'], identifier=EX_NS['use5'], time=datetime.datetime.now())
    use.add_attributes([(PROV_ROLE, 'somerole')])
    self.add_types(use)
    self.add_locations(use)
    self.add_labels(use)
    self.add_further_attributes(use)
    self.do_tests(document)