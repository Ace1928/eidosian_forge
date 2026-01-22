from docutils import nodes
from sphinx.locale import admonitionlabels
from sphinx.writers.html5 import HTML5Translator as SphinxHTML5Translator
def visit_admonition(self, node, name=''):
    """Uses the h3 tag for admonition titles instead of the p tag."""
    self.body.append(self.starttag(node, 'div', CLASS='admonition ' + name))
    if name:
        title = f"<h3 class='admonition-title'> {admonitionlabels[name]}</h3>"
        self.body.append(title)