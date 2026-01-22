import ast
import collections
import os
import re
import shutil
import sys
import tempfile
import traceback
import pasta
def process_tree(self, root_directory, output_root_directory, copy_other_files):
    """Processes upgrades on an entire tree of python files in place.

    Note that only Python files. If you have custom code in other languages,
    you will need to manually upgrade those.

    Args:
      root_directory: Directory to walk and process.
      output_root_directory: Directory to use as base.
      copy_other_files: Copy files that are not touched by this converter.

    Returns:
      A tuple of files processed, the report string for all files, and a dict
        mapping filenames to errors encountered in that file.
    """
    if output_root_directory == root_directory:
        return self.process_tree_inplace(root_directory)
    if output_root_directory and os.path.exists(output_root_directory):
        print('Output directory %r must not already exist.' % output_root_directory)
        sys.exit(1)
    norm_root = os.path.split(os.path.normpath(root_directory))
    norm_output = os.path.split(os.path.normpath(output_root_directory))
    if norm_root == norm_output:
        print('Output directory %r same as input directory %r' % (root_directory, output_root_directory))
        sys.exit(1)
    files_to_process = []
    files_to_copy = []
    for dir_name, _, file_list in os.walk(root_directory):
        py_files = [f for f in file_list if f.endswith('.py')]
        copy_files = [f for f in file_list if not f.endswith('.py')]
        for filename in py_files:
            fullpath = os.path.join(dir_name, filename)
            fullpath_output = os.path.join(output_root_directory, os.path.relpath(fullpath, root_directory))
            files_to_process.append((fullpath, fullpath_output))
        if copy_other_files:
            for filename in copy_files:
                fullpath = os.path.join(dir_name, filename)
                fullpath_output = os.path.join(output_root_directory, os.path.relpath(fullpath, root_directory))
                files_to_copy.append((fullpath, fullpath_output))
    file_count = 0
    tree_errors = {}
    report = ''
    report += '=' * 80 + '\n'
    report += 'Input tree: %r\n' % root_directory
    report += '=' * 80 + '\n'
    for input_path, output_path in files_to_process:
        output_directory = os.path.dirname(output_path)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        if os.path.islink(input_path):
            link_target = os.readlink(input_path)
            link_target_output = os.path.join(output_root_directory, os.path.relpath(link_target, root_directory))
            if (link_target, link_target_output) in files_to_process:
                os.symlink(link_target_output, output_path)
            else:
                report += 'Copying symlink %s without modifying its target %s' % (input_path, link_target)
                os.symlink(link_target, output_path)
            continue
        file_count += 1
        _, l_report, l_errors = self.process_file(input_path, output_path)
        tree_errors[input_path] = l_errors
        report += l_report
    for input_path, output_path in files_to_copy:
        output_directory = os.path.dirname(output_path)
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
        shutil.copy(input_path, output_path)
    return (file_count, report, tree_errors)