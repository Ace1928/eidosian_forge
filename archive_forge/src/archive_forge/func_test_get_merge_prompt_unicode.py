from .. import errors, mail_client, osutils, tests, urlutils
def test_get_merge_prompt_unicode(self):
    """Prompt, to and subject are unicode, the attachement is binary"""
    editor = mail_client.Editor(None)
    prompt = editor._get_merge_prompt('fooሴ', 'barሴ', 'bazሴ', 'quxሴ'.encode())
    self.assertContainsRe(prompt, 'fooሴ(.|\n)*barሴ(.|\n)*bazሴ(.|\n)*quxሴ')
    editor._get_merge_prompt('foo', 'bar', 'baz', b'qux\xff')