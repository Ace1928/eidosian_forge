import argparse
from gettext import gettext
import os
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import final
from typing import List
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import _pytest._io
from _pytest.config.exceptions import UsageError
from _pytest.deprecated import check_ispytest
def parse_known_and_unknown_args(self, args: Sequence[Union[str, 'os.PathLike[str]']], namespace: Optional[argparse.Namespace]=None) -> Tuple[argparse.Namespace, List[str]]:
    """Parse the known arguments at this point, and also return the
        remaining unknown arguments.

        :returns:
            A tuple containing an argparse namespace object for the known
            arguments, and a list of the unknown arguments.
        """
    optparser = self._getparser()
    strargs = [os.fspath(x) for x in args]
    return optparser.parse_known_args(strargs, namespace=namespace)