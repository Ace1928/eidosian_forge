from prov.model import *
def test_activity_6(self):
    document = self.new_document()
    a = document.activity(EX_NS['a6'])
    a.add_attributes([(PROV_LABEL, 'activity6')])
    self.add_locations(a)
    self.do_tests(document)