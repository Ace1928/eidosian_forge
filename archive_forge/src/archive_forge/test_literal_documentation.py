from __future__ import print_function
import pytest  # NOQA
from .roundtrip import YAML  # does an automatic dedent on load

YAML 1.0 allowed root level literal style without indentation:
  "Usually top level nodes are not indented" (example 4.21 in 4.6.3)
YAML 1.1 is a bit vague but says:
  "Regardless of style, scalar content must always be indented by at least one space"
  (4.4.3)
  "In general, the document’s node is indented as if it has a parent indented at -1 spaces."
  (4.3.3)
YAML 1.2 is again clear about root literal level scalar after directive in example 9.5:

%YAML 1.2
--- |
%!PS-Adobe-2.0
...
%YAML1.2
---
# Empty
...
