import os
import re
from parso import python_bytes_to_unicode
from jedi.debug import dbg
from jedi.file_io import KnownContentFileIO, FolderIO
from jedi.inference.names import SubModuleName
from jedi.inference.imports import load_module_from_path
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.gradual.conversion import convert_names
def recurse_find_python_folders_and_files(folder_io, except_paths=()):
    except_paths = set(except_paths)
    except_paths_relative = set()
    for root_folder_io, folder_ios, file_ios in folder_io.walk():
        for file_io in file_ios:
            path = file_io.path
            if path.suffix in ('.py', '.pyi'):
                if path not in except_paths:
                    yield (None, file_io)
            if path.name == '.gitignore':
                ignored_paths_abs, ignored_paths_rel = gitignored_paths(root_folder_io, file_io)
                except_paths |= ignored_paths_abs
                except_paths_relative |= ignored_paths_rel
        except_paths_relative_expanded = expand_relative_ignore_paths(root_folder_io, except_paths_relative)
        folder_ios[:] = [folder_io for folder_io in folder_ios if folder_io.path not in except_paths and folder_io.path not in except_paths_relative_expanded and (folder_io.get_base_name() not in _IGNORE_FOLDERS)]
        for folder_io in folder_ios:
            yield (folder_io, None)