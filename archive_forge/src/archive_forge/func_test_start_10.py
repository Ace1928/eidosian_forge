from prov.model import *
def test_start_10(self):
    document = self.new_document()
    start = document.start(EX_NS['a1'], starter=EX_NS['a2'], time=datetime.datetime.now())
    start.add_attributes([(PROV_ROLE, 'egg-cup'), (PROV_ROLE, 'boiling-water')])
    self.add_types(start)
    self.add_locations(start)
    self.add_labels(start)
    self.add_further_attributes(start)
    self.do_tests(document)