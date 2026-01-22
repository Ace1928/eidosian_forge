from ._authorizer import ClosedAuthorizer
from ._checker import Checker
import macaroonbakery.checkers as checkers
from ._oven import Oven
@property
def oven(self):
    return self._oven