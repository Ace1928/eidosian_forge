from prov.model import ProvDocument
def test_flattening_2_bundles_with_default_namespace(self):
    prov_doc = document_with_n_bundles_having_default_namespace(2)
    flattened = prov_doc.flattened()
    self.do_tests(flattened)