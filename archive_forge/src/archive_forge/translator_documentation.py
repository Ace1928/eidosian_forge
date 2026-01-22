from docutils import nodes
from sphinx.locale import admonitionlabels
from sphinx.writers.html5 import HTML5Translator as SphinxHTML5Translator
Visit a paragraph HTML element.

        Replaces implicit headings with an h3 tag and defers to default
        behavior for normal paragraph elements.
        