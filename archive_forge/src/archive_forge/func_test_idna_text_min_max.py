@given(data())
def test_idna_text_min_max(self, data):
    """
            idna_text() raises AssertionError if min_size is < 1.
            """
    self.assertRaises(AssertionError, data.draw, idna_text(min_size=0))
    self.assertRaises(AssertionError, data.draw, idna_text(max_size=0))