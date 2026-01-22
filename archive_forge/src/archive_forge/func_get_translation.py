import locale
from gettext import NullTranslations, translation
from os import path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
def get_translation(catalog: str, namespace: str='general') -> Callable[[str], str]:
    """Get a translation function based on the *catalog* and *namespace*.

    The extension can use this API to translate the messages on the
    extension::

        import os
        from sphinx.locale import get_translation

        MESSAGE_CATALOG_NAME = 'myextension'  # name of *.pot, *.po and *.mo files
        _ = get_translation(MESSAGE_CATALOG_NAME)
        text = _('Hello Sphinx!')


        def setup(app):
            package_dir = os.path.abspath(os.path.dirname(__file__))
            locale_dir = os.path.join(package_dir, 'locales')
            app.add_message_catalog(MESSAGE_CATALOG_NAME, locale_dir)

    With this code, sphinx searches a message catalog from
    ``${package_dir}/locales/${language}/LC_MESSAGES/myextension.mo``.
    The :confval:`language` is used for the searching.

    .. versionadded:: 1.8
    """

    def gettext(message: str, *args: Any) -> str:
        if not is_translator_registered(catalog, namespace):
            return _TranslationProxy(_lazy_translate, catalog, namespace, message)
        else:
            translator = get_translator(catalog, namespace)
            if len(args) <= 1:
                return translator.gettext(message)
            else:
                return translator.ngettext(message, args[0], args[1])
    return gettext