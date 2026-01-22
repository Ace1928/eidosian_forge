from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
def user_set_password(self, user, password):
    """
        Passlib replacement for User.set_password()
        """
    if password is None:
        user.set_unusable_password()
    else:
        cat = self.get_user_category(user)
        user.password = self.context.hash(password, category=cat)