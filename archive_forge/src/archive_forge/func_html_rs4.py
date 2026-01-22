import jinja2  # type: ignore
from rpy2.robjects import (vectors,
from rpy2 import rinterface
from rpy2.robjects.packages import SourceCode
from rpy2.robjects.packages import wherefrom
from IPython import get_ipython  # type: ignore
def html_rs4(obj, table_class='rpy2_table'):
    d = _dict_ridentifiedobject(obj)
    d['table_class'] = table_class
    html = template_rs4.render(d)
    return html