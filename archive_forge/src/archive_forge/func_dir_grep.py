import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def dir_grep(tree, path, relpath, opts, revno, path_prefix):
    rpath = relpath
    if relpath:
        rpath = osutils.pathjoin('..', relpath)
    from_dir = osutils.pathjoin(relpath, path)
    if opts.from_root:
        from_dir = None
        recursive = True
    to_grep = []
    to_grep_append = to_grep.append
    outputter = opts.outputter
    for fp, fc, fkind, entry in tree.list_files(include_root=False, from_dir=from_dir, recursive=opts.recursive):
        if _skip_file(opts.include, opts.exclude, fp):
            continue
        if fc == 'V' and fkind == 'file':
            tree_path = osutils.pathjoin(from_dir if from_dir else '', fp)
            if revno is not None:
                cache_id = tree.get_file_revision(tree_path)
                if cache_id in outputter.cache:
                    outputter.write_cached_lines(cache_id, revno)
                else:
                    to_grep_append((tree_path, (fp, tree_path)))
            else:
                if from_dir is None:
                    from_dir = '.'
                path_for_file = osutils.pathjoin(tree.basedir, from_dir, fp)
                if opts.files_with_matches or opts.files_without_match:
                    with open(path_for_file, 'rb', buffering=4096) as file:
                        _file_grep_list_only_wtree(file, fp, opts, path_prefix)
                else:
                    with open(path_for_file, 'rb') as f:
                        _file_grep(f.read(), fp, opts, revno, path_prefix)
    if revno is not None:
        for (path, tree_path), chunks in tree.iter_files_bytes(to_grep):
            path = _make_display_path(relpath, path)
            _file_grep(b''.join(chunks), path, opts, revno, path_prefix, tree.get_file_revision(tree_path))