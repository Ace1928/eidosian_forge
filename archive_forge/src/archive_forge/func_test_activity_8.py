from prov.model import *
def test_activity_8(self):
    document = self.new_document()
    a = document.activity(EX_NS['a8'], startTime=datetime.datetime.now(), endTime=datetime.datetime.now())
    a.add_attributes([(PROV_LABEL, 'activity8')])
    self.add_types(a)
    self.add_types(a)
    self.add_locations(a)
    self.add_locations(a)
    self.add_labels(a)
    self.add_labels(a)
    self.do_tests(document)