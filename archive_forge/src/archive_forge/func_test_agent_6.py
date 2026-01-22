from prov.model import *
def test_agent_6(self):
    document = self.new_document()
    a = document.agent(EX_NS['ag6'])
    a.add_attributes([(PROV_LABEL, 'agent6')])
    self.add_types(a)
    self.do_tests(document)