import docutils.parsers
import docutils.statemachine
from docutils.parsers.rst import states
from docutils import frontend, nodes, Component
from docutils.transforms import universal
def convert_directive_function(directive_fn):
    """
    Define & return a directive class generated from `directive_fn`.

    `directive_fn` uses the old-style, functional interface.
    """

    class FunctionalDirective(Directive):
        option_spec = getattr(directive_fn, 'options', None)
        has_content = getattr(directive_fn, 'content', False)
        _argument_spec = getattr(directive_fn, 'arguments', (0, 0, False))
        required_arguments, optional_arguments, final_argument_whitespace = _argument_spec

        def run(self):
            return directive_fn(self.name, self.arguments, self.options, self.content, self.lineno, self.content_offset, self.block_text, self.state, self.state_machine)
    return FunctionalDirective