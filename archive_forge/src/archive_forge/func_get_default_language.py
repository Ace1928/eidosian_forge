from enchant.errors import Error
import locale
def get_default_language(default=None):
    """Determine the user's default language, if possible.

    This function uses the 'locale' module to try to determine
    the user's preferred language.  The return value is as
    follows:

        * if a locale is available for the LC_MESSAGES category,
          that language is used
        * if a default locale is available, that language is used
        * if the keyword argument <default> is given, it is used
        * if nothing else works, None is returned

    Note that determining the user's language is in general only
    possible if they have set the necessary environment variables
    on their system.
    """
    try:
        tag = locale.getlocale()[0]
        if tag is None:
            tag = locale.getdefaultlocale()[0]
            if tag is None:
                raise Error('No default language available')
        return tag
    except Exception:
        pass
    return default