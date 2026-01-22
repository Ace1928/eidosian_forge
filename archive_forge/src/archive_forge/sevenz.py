#
# diffoscope: in-depth comparison of files, archives, and directories
#
# Copyright © 2024 Chris Lamb <lamby@debian.org>
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

import os
import re
import logging
import subprocess

from diffoscope.exc import RequiredToolNotFound
from diffoscope.tools import tool_required
from diffoscope.tempfiles import get_temporary_directory
from diffoscope.difference import Difference

from .utils.archive import Archive
from .utils.file import File
from .utils.command import Command, our_check_output

logger = logging.getLogger(__name__)


class SevenZList(Command):
    @tool_required("7z")
    def cmdline(self):
        return (
            "7z",
            "l",
            self.path,
        )

    def filter(self, line):
        val = line.decode("utf-8")
        if val.startswith("Listing archive: ") or val.startswith("Path = "):
            return b""
        return line


class SevenZContainer(Archive):
    @tool_required("7z")
    def open_archive(self):
        self._temp_dir = get_temporary_directory(suffix="7z")

        try:
            our_check_output(
                ("7z", "e", os.path.abspath(self.source.path)),
                cwd=self._temp_dir.name,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return False

        return self

    def close_archive(self):
        self._temp_dir.cleanup()

    def get_member_names(self):
        return os.listdir(self._temp_dir.name)

    def extract(self, member_name, dest_dir):
        return os.path.join(self._temp_dir.name, member_name)


class SevenZFile(File):
    DESCRIPTION = "Filesystem image"
    FILE_TYPE_RE = re.compile(r"^DOS/MBR boot sector;")
    CONTAINER_CLASSES = [SevenZContainer]

    def compare_details(self, other, source=None):
        return [
            Difference.from_operation(
                SevenZList, self.path, other.path, source="7z l"
            )
        ]

    @classmethod
    def recognizes(cls, file):
        if not super().recognizes(file):
            return False

        try:
            cmd = SevenZList(file.path)
            cmd.start()
        except RequiredToolNotFound:
            return False

        return b"Type = gzip\n" not in cmd.output
