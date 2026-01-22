from kivy.tests.common import GraphicUnitTest

    def test_default_label(self):
        from kivy.uix.label import Label
        self.render(Label())

    def test_button_state_down(self):
        from kivy.uix.button import Button
        self.render(Button(state='down'))

    def test_label_text(self):
        from kivy.uix.label import Label
        self.render(Label(text='Hello world'))

    def test_label_font_size(self):
        from kivy.uix.label import Label
        self.render(Label(text='Hello world', font_size=16))

    def test_label_font_size(self):
        from kivy.uix.label import Label
        self.render(Label(text='Hello world'))
    