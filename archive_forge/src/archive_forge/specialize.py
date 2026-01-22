#
# diffoscope: in-depth comparison of files, archives, and directories
#
# Copyright © 2016-2020, 2023 Chris Lamb <lamby@debian.org>
#
# diffoscope is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# diffoscope is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with diffoscope.  If not, see <https://www.gnu.org/licenses/>.

import logging

from diffoscope.profiling import profile
from diffoscope.comparators.text import TextFile

from ...utils import format_class

from .. import ComparatorManager

logger = logging.getLogger(__name__)


def try_recognize(file, cls, recognizes):
    # Does this file class match?
    with profile("recognizes", file):
        # logger.debug("trying %s on %s", cls, file)
        if not recognizes(file):
            return False

    # Found a match; perform type magic
    logger.debug(
        "Using %s for %s",
        format_class(cls, strip="diffoscope.comparators."),
        file.name,
    )
    specialize_as(cls, file)

    return True


def specialize_as(cls, file):
    """
    Sometimes it is near-certain that files within a Container with a given
    extension (say) are of a known File type. We therefore do not need to run
    libmagic on these files. Using this facily within a Container can be an
    optimization, especially in cases where the Container contains hundreds of
    similar/small files. (This can be seen in the case of apktool and .smali
    files). In this case, this method can be used to essentially fix/force the
    type. Care should naturally be taken within Container implementations;
    such as checking the file extension and so forth.
    """

    new_cls = type(cls.__name__, (cls, type(file)), {})
    file.__class__ = new_cls
    return file


def specialize(file):
    # If we already know the class (ie. via `specialize_as`), then we do not
    # need to run `File.recognizes` at all.
    for cls in ComparatorManager().classes:
        if isinstance(file, cls):
            return file

    # Run the usual `File.recognizes` implementation.
    for cls in ComparatorManager().classes:
        if try_recognize(file, cls, cls.recognizes):
            return file

    # If there are no matches, run the fallback implementation.
    for cls in ComparatorManager().classes:
        if try_recognize(file, cls, cls.fallback_recognizes):
            logger.debug(
                "File recognized by fallback. Magic says: %s",
                file.magic_file_type,
            )
            return file

    logger.debug(
        "%s not identified by any comparator. Magic says: %s",
        file.name,
        file.magic_file_type,
    )

    if file.magic_mime_type == "text/plain":
        logger.debug(
            "However, %s is probably text; using %s",
            file.name,
            format_class(TextFile, strip="diffoscope.comparators."),
        )
        return specialize_as(TextFile, file)

    return file


def is_direct_instance(file, cls):
    # is_direct_instance(<GzipFile>, GzipFile) == True, but
    # is_direct_instance(<IpkFile>, GzipFile) == False
    if not isinstance(file, cls):
        return False
    for c in ComparatorManager().classes:
        if c is not cls and isinstance(file, c):
            return False
    return True
