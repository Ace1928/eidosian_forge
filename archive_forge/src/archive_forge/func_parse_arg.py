import argparse
import os
from time import time
from pyzstd import compress_stream, decompress_stream, \
def parse_arg():
    p = argparse.ArgumentParser(prog='CLI of pyzstd module', description="The command style is similar to zstd's CLI, but there are some differences.\nZstd's CLI should be faster, it has some I/O optimizations.", epilog='Examples of use:\n  compress a file:\n    python -m pyzstd -c IN_FILE -o OUT_FILE\n  decompress a file:\n    python -m pyzstd -d IN_FILE -o OUT_FILE\n  create a tar archive:\n    python -m pyzstd --tar-input-dir DIR -o OUT_FILE\n  extract a tar archive, output will forcibly overwrite existing files:\n    python -m pyzstd -d IN_FILE --tar-output-dir DIR\n  train a zstd dictionary, ** traverses sub-directories:\n    python -m pyzstd --train "E:\\cpython\\**\\*.c" -o OUT_FILE', formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group('Common arguments')
    g.add_argument('-D', '--dict', metavar='FILE', type=argparse.FileType('rb'), help='use FILE as zstd dictionary for compression or decompression')
    g.add_argument('-o', '--output', metavar='FILE', type=str, help='result stored into FILE')
    g.add_argument('-f', action='store_true', help='disable output check, allows overwriting existing file.')
    g = p.add_argument_group('Compression arguments')
    gm = g.add_mutually_exclusive_group()
    gm.add_argument('-c', '--compress', metavar='FILE', type=str, help='compress FILE')
    gm.add_argument('--tar-input-dir', metavar='DIR', type=str, help='create a tar archive from DIR. this option overrides -c/--compress option.')
    g.add_argument('-l', '--level', metavar='#', default=compressionLevel_values.default, action=range_action(compressionLevel_values.min, compressionLevel_values.max), help='compression level, range: [{},{}], default: {}.'.format(compressionLevel_values.min, compressionLevel_values.max, compressionLevel_values.default))
    g.add_argument('-t', '--threads', metavar='#', default=0, action=range_action(*CParameter.nbWorkers.bounds(), True), help='spawns # threads to compress. if this option is not specified or is 0, use single thread mode.')
    g.add_argument('--long', metavar='#', nargs='?', const=27, default=-1, action=range_action(*CParameter.windowLog.bounds(), True), help='enable long distance matching with given windowLog (default #: 27)')
    g.add_argument('--no-checksum', action='store_false', dest='checksum', default=True, help="don't add 4-byte XXH64 checksum to the frame")
    g.add_argument('--no-dictID', action='store_false', dest='write_dictID', default=True, help="don't write dictID into frame header (dictionary compression only)")
    g = p.add_argument_group('Decompression arguments')
    gm = g.add_mutually_exclusive_group()
    gm.add_argument('-d', '--decompress', metavar='FILE', type=str, help='decompress FILE')
    g.add_argument('--tar-output-dir', metavar='DIR', type=str, help='extract tar archive to DIR, output will forcibly overwrite existing files. this option overrides -o/--output option.')
    gm.add_argument('--test', metavar='FILE', type=str, help='try to decompress FILE to check integrity')
    g.add_argument('--windowLogMax', metavar='#', default=0, action=range_action(*DParameter.windowLogMax.bounds(), True), help='set a memory usage limit for decompression (windowLogMax)')
    g = p.add_argument_group('Dictionary builder')
    g.add_argument('--train', metavar='GLOB_PATH', type=str, help='create a dictionary from a training set of files')
    g.add_argument('--maxdict', metavar='SIZE', type=int, default=112640, help='limit dictionary to SIZE bytes (default: 112640)')
    g.add_argument('--dictID', metavar='DICT_ID', default=None, action=range_action(1, 4294967295), help='specify dictionary ID value (default: random)')
    args = p.parse_args()
    if args.compress is not None:
        args.input = open(args.compress, 'rb', buffering=C_READ_BUFFER)
    elif args.decompress is not None:
        args.input = open(args.decompress, 'rb', buffering=D_READ_BUFFER)
    elif args.test is not None:
        args.input = open(args.test, 'rb', buffering=D_READ_BUFFER)
    else:
        args.input = None
    if args.output is not None:
        open_output(args, args.output)
    if args.dict is not None:
        zd_content = args.dict.read()
        args.dict.close()
        is_raw = zd_content[:4] != b'7\xa40\xec'
        args.zd = ZstdDict(zd_content, is_raw)
    else:
        args.zd = None
    functions = [args.compress, args.decompress, args.test, args.train, args.tar_input_dir]
    if sum((1 for i in functions if i is not None)) > 1:
        raise Exception('Wrong arguments combination')
    return args