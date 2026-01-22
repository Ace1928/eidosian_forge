@given(data())
@settings(max_examples=10)
def test_hostname_labels_long_idn_punycode(self, data):
    """
            hostname_labels() handles case where idna_text() generates text
            that encoded to punycode ends up as longer than allowed.
            """

    @composite
    def mock_idna_text(draw, min_size, max_size):
        return u'á' * max_size
    with patch('hyperlink.hypothesis.idna_text', mock_idna_text):
        label = data.draw(hostname_labels())
        try:
            check_label(label)
            idna_encode(label)
        except UnicodeError:
            raise AssertionError('Invalid IDN label: {!r}'.format(label))