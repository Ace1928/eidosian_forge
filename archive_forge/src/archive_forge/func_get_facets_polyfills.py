import base64
import sys
from typing import Iterable, Tuple, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from mlflow.exceptions import MlflowException
from mlflow.protos import facet_feature_statistics_pb2
from mlflow.recipes.cards import histogram_generator
def get_facets_polyfills() -> str:
    """
    A JS polyfill/monkey-patching function that fixes issue where objectURL passed as a
    "base" argument to the URL constructor ends up in a "invalid URL" exception.

    Polymer is using parent's URL in its internal asset URL resolution system, while MLFLow
    artifact rendering engine uses object URLs to display iframed artifacts code. This ends up
    in object URL being used in `new URL()` constructor which needs to be patched.

    Original function code:

    (function patchURLConstructor() {
        const _originalURLConstructor = window.URL;
        window.URL = function (url, base) {
            if (typeof base === "string" && base.startsWith("blob:")) {
                return new URL(base);
            }
            return new _originalURLConstructor(url, base);
        };
    })();
    """
    return '\n!function() {\n  let t = window.URL;\n  window.URL = function(n, e) {\n    if (typeof e === "string" && e.startsWith("blob:")) {\n      return new URL(e);\n    } else {\n      return new t(n, e);\n    }\n  }\n}();\n'