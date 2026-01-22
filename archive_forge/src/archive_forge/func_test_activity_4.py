from prov.model import *
def test_activity_4(self):
    document = self.new_document()
    a = document.activity(EX_NS['a4'])
    a.add_attributes([(PROV_LABEL, 'activity4')])
    self.add_labels(a)
    self.do_tests(document)