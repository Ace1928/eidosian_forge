from __future__ import unicode_literals
def stable_wrap(wrapper, paragraph_text):
    """
  textwrap doesn't appear to be stable. We run it multiple times until it
  converges
  """
    history = [paragraph_text]
    prev_text = paragraph_text
    wrap_text = paragraph_text
    for _ in range(8):
        lines = wrapper.wrap(wrap_text)
        next_text = '\n'.join((line for line in lines))
        if next_text == prev_text:
            return lines
        prev_text = next_text
        history.append(next_text)
        lines[0] = lines[0][len(wrapper.initial_indent):]
        lines[1:] = [line[len(wrapper.subsequent_indent):] for line in lines[1:]]
        wrap_text = '\n'.join(lines)
    assert False, 'textwrap failed to converge on:\n\n {}'.format('\n\n'.join(history))
    return []