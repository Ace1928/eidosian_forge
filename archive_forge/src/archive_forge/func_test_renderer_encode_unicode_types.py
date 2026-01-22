from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import Renderer
def test_renderer_encode_unicode_types(self):
    mime_types = ['image/svg+xml', 'text/html', 'text/json']
    for mime in mime_types:
        info = {'mime_type': mime}
        encoded = Renderer.encode(('Testing «ταБЬℓσ»: 1<2 & 4+1>3', info))
        self.assertTrue(isinstance(encoded, bytes))