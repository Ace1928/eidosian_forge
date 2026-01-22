import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def get_tokens_unprocessed(self, text):
    dylexer = DylanLexer(**self.options)
    curcode = ''
    insertions = []
    for match in self._line_re.finditer(text):
        line = match.group()
        m = self._prompt_re.match(line)
        if m is not None:
            end = m.end()
            insertions.append((len(curcode), [(0, Generic.Prompt, line[:end])]))
            curcode += line[end:]
        else:
            if curcode:
                for item in do_insertions(insertions, dylexer.get_tokens_unprocessed(curcode)):
                    yield item
                curcode = ''
                insertions = []
            yield (match.start(), Generic.Output, line)
    if curcode:
        for item in do_insertions(insertions, dylexer.get_tokens_unprocessed(curcode)):
            yield item