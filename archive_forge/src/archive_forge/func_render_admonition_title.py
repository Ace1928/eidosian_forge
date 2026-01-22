from ._base import DirectivePlugin
def render_admonition_title(self, text):
    return '<p class="admonition-title">' + text + '</p>\n'