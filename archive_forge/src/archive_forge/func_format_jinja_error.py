from .compat import escape
from .jsonify import encode
def format_jinja_error(exc_value):
    """
        Implements ``Jinja`` renderer error formatting.
        """
    retval = "<h4>Jinja2 error in '%s' on line %d</h4><div>%s</div>"
    if isinstance(exc_value, jTemplateSyntaxError):
        retval = retval % (exc_value.name, exc_value.lineno, exc_value.message)
        retval += format_line_context(exc_value.filename, exc_value.lineno)
        return retval