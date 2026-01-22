from zope.interface import implementer
from twisted import plugin
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.strcred import ICheckerFactory
from twisted.internet import defer
def checkSpwd(self, spwd, username, password):
    """
        Obtain the encrypted password for C{username} from the
        Unix shadow password database using L{spwd.getspnam},
        and see if it it matches it matches C{password}.

        @param spwd: Module which provides functions which
                     access to the Unix shadow password database.
        @type spwd: C{module}
        @param username: The user to look up in the Unix password database.
        @type username: L{unicode}/L{str} or L{bytes}
        @param password: The password to compare.
        @type username: L{unicode}/L{str} or L{bytes}
        """
    try:
        if isinstance(username, bytes):
            username = username.decode('utf-8')
        if getattr(spwd.struct_spwd, 'sp_pwdp', None):
            cryptedPass = spwd.getspnam(username).sp_pwdp
        else:
            cryptedPass = spwd.getspnam(username).sp_pwd
    except KeyError:
        return defer.fail(UnauthorizedLogin())
    else:
        if verifyCryptedPassword(cryptedPass, password):
            return defer.succeed(username)