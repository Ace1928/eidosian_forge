from kivy.tests.common import GraphicUnitTest
def test_clipboard_not_dummy(self):
    clippy = self._clippy
    if clippy.__class__.__name__ == 'ClipboardDummy':
        self.fail('Something went wrong "dummy" clipboard is being used')