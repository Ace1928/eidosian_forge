from prov.model import *
def test_agent_3(self):
    document = self.new_document()
    a = document.agent(EX_NS['ag3'])
    a.add_attributes([(PROV_LABEL, 'agent3'), (PROV_LABEL, Literal('hello'))])
    self.do_tests(document)