import six
import __internals
def input_with_prefill(prompt, text):

    def hook():
        readline.insert_text(text)
        readline.redisplay()
    readline.set_pre_input_hook(hook)
    result = six.moves.input(prompt)
    readline.set_pre_input_hook()
    return result