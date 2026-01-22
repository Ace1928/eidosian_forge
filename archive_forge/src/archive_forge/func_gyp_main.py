import copy
import gyp.input
import argparse
import os.path
import re
import shlex
import sys
import traceback
from gyp.common import GypError
def gyp_main(args):
    my_name = os.path.basename(sys.argv[0])
    usage = 'usage: %(prog)s [options ...] [build_file ...]'
    parser = RegeneratableOptionParser(usage=usage.replace('%s', '%(prog)s'))
    parser.add_argument('--build', dest='configs', action='append', help='configuration for build after project generation')
    parser.add_argument('--check', dest='check', action='store_true', help='check format of gyp files')
    parser.add_argument('--config-dir', dest='config_dir', action='store', env_name='GYP_CONFIG_DIR', default=None, help='The location for configuration files like include.gypi.')
    parser.add_argument('-d', '--debug', dest='debug', metavar='DEBUGMODE', action='append', default=[], help='turn on a debugging mode for debugging GYP.  Supported modes are "variables", "includes" and "general" or "all" for all of them.')
    parser.add_argument('-D', dest='defines', action='append', metavar='VAR=VAL', env_name='GYP_DEFINES', help='sets variable VAR to value VAL')
    parser.add_argument('--depth', dest='depth', metavar='PATH', type='path', help='set DEPTH gyp variable to a relative path to PATH')
    parser.add_argument('-f', '--format', dest='formats', action='append', env_name='GYP_GENERATORS', regenerate=False, help='output formats to generate')
    parser.add_argument('-G', dest='generator_flags', action='append', default=[], metavar='FLAG=VAL', env_name='GYP_GENERATOR_FLAGS', help='sets generator flag FLAG to VAL')
    parser.add_argument('--generator-output', dest='generator_output', action='store', default=None, metavar='DIR', type='path', env_name='GYP_GENERATOR_OUTPUT', help='puts generated build files under DIR')
    parser.add_argument('--ignore-environment', dest='use_environment', action='store_false', default=True, regenerate=False, help='do not read options from environment variables')
    parser.add_argument('-I', '--include', dest='includes', action='append', metavar='INCLUDE', type='path', help='files to include in all loaded .gyp files')
    parser.add_argument('--no-circular-check', dest='circular_check', action='store_false', default=True, regenerate=False, help="don't check for circular relationships between files")
    parser.add_argument('--no-parallel', action='store_true', default=False, help='Disable multiprocessing')
    parser.add_argument('-S', '--suffix', dest='suffix', default='', help='suffix to add to generated files')
    parser.add_argument('--toplevel-dir', dest='toplevel_dir', action='store', default=None, metavar='DIR', type='path', help='directory to use as the root of the source tree')
    parser.add_argument('-R', '--root-target', dest='root_targets', action='append', metavar='TARGET', help='include only TARGET and its deep dependencies')
    options, build_files_arg = parser.parse_args(args)
    build_files = build_files_arg
    if not options.config_dir:
        home = None
        home_dot_gyp = None
        if options.use_environment:
            home_dot_gyp = os.environ.get('GYP_CONFIG_DIR', None)
            if home_dot_gyp:
                home_dot_gyp = os.path.expanduser(home_dot_gyp)
        if not home_dot_gyp:
            home_vars = ['HOME']
            if sys.platform in ('cygwin', 'win32'):
                home_vars.append('USERPROFILE')
            for home_var in home_vars:
                home = os.getenv(home_var)
                if home:
                    home_dot_gyp = os.path.join(home, '.gyp')
                    if not os.path.exists(home_dot_gyp):
                        home_dot_gyp = None
                    else:
                        break
    else:
        home_dot_gyp = os.path.expanduser(options.config_dir)
    if home_dot_gyp and (not os.path.exists(home_dot_gyp)):
        home_dot_gyp = None
    if not options.formats:
        generate_formats = []
        if options.use_environment:
            generate_formats = os.environ.get('GYP_GENERATORS', [])
        if generate_formats:
            generate_formats = re.split('[\\s,]', generate_formats)
        if generate_formats:
            options.formats = generate_formats
        elif sys.platform == 'darwin':
            options.formats = ['xcode']
        elif sys.platform in ('win32', 'cygwin'):
            options.formats = ['msvs']
        else:
            options.formats = ['make']
    if not options.generator_output and options.use_environment:
        g_o = os.environ.get('GYP_GENERATOR_OUTPUT')
        if g_o:
            options.generator_output = g_o
    options.parallel = not options.no_parallel
    for mode in options.debug:
        gyp.debug[mode] = 1
    if DEBUG_GENERAL in gyp.debug:
        DebugOutput(DEBUG_GENERAL, 'running with these options:')
        for option, value in sorted(options.__dict__.items()):
            if option[0] == '_':
                continue
            if isinstance(value, str):
                DebugOutput(DEBUG_GENERAL, "  %s: '%s'", option, value)
            else:
                DebugOutput(DEBUG_GENERAL, '  %s: %s', option, value)
    if not build_files:
        build_files = FindBuildFiles()
    if not build_files:
        raise GypError((usage + '\n\n%s: error: no build_file') % (my_name, my_name))
    if not options.depth:
        for build_file in build_files:
            build_file_dir = os.path.abspath(os.path.dirname(build_file))
            build_file_dir_components = build_file_dir.split(os.path.sep)
            components_len = len(build_file_dir_components)
            for index in range(components_len - 1, -1, -1):
                if build_file_dir_components[index] == 'src':
                    options.depth = os.path.sep.join(build_file_dir_components)
                    break
                del build_file_dir_components[index]
            if options.depth:
                break
        if not options.depth:
            raise GypError('Could not automatically locate src directory.  This isa temporary Chromium feature that will be removed.  Use--depth as a workaround.')
    if not options.toplevel_dir:
        options.toplevel_dir = options.depth
    cmdline_default_variables = {}
    defines = []
    if options.use_environment:
        defines += ShlexEnv('GYP_DEFINES')
    if options.defines:
        defines += options.defines
    cmdline_default_variables = NameValueListToDict(defines)
    if DEBUG_GENERAL in gyp.debug:
        DebugOutput(DEBUG_GENERAL, 'cmdline_default_variables: %s', cmdline_default_variables)
    includes = []
    if home_dot_gyp:
        default_include = os.path.join(home_dot_gyp, 'include.gypi')
        if os.path.exists(default_include):
            print('Using overrides found in ' + default_include)
            includes.append(default_include)
    if options.includes:
        includes.extend(options.includes)
    gen_flags = []
    if options.use_environment:
        gen_flags += ShlexEnv('GYP_GENERATOR_FLAGS')
    if options.generator_flags:
        gen_flags += options.generator_flags
    generator_flags = NameValueListToDict(gen_flags)
    if DEBUG_GENERAL in gyp.debug.keys():
        DebugOutput(DEBUG_GENERAL, 'generator_flags: %s', generator_flags)
    for format in set(options.formats):
        params = {'options': options, 'build_files': build_files, 'generator_flags': generator_flags, 'cwd': os.getcwd(), 'build_files_arg': build_files_arg, 'gyp_binary': sys.argv[0], 'home_dot_gyp': home_dot_gyp, 'parallel': options.parallel, 'root_targets': options.root_targets, 'target_arch': cmdline_default_variables.get('target_arch', '')}
        [generator, flat_list, targets, data] = Load(build_files, format, cmdline_default_variables, includes, options.depth, params, options.check, options.circular_check)
        generator.GenerateOutput(flat_list, targets, data, params)
        if options.configs:
            valid_configs = targets[flat_list[0]]['configurations']
            for conf in options.configs:
                if conf not in valid_configs:
                    raise GypError('Invalid config specified via --build: %s' % conf)
            generator.PerformBuild(data, options.configs, params)
    return 0