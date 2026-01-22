import yaql.tests
def test_split_on_string(self):
    self.assertEqual(['Words', 'words', 'words', ''], self.eval("'Words, words, words.'.split(regex(`\\W+`))"))
    self.assertEqual(['Words', ', ', 'words', ', ', 'words', '.', ''], self.eval("'Words, words, words.'.split(regex(`(\\W+)`))"))
    self.assertEqual(['Words', 'words, words.'], self.eval("'Words, words, words.'.split(regex(`\\W+`), 1)"))
    self.assertEqual(['0', '3', '9'], self.eval("'0a3B9'.split(regex('[a-f]+', ignoreCase => true))"))