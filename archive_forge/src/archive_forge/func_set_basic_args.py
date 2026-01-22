import argparse
import codecs
import srt
import logging
import sys
import itertools
import os
def set_basic_args(args):
    if getattr(args, 'inplace', None):
        if args.input == DASH_STREAM_MAP['input']:
            raise ValueError('Cannot use --inplace on stdin')
        if args.output != DASH_STREAM_MAP['output']:
            raise ValueError('Cannot use -o and -p together')
        args.output = args.input
    for stream_name in ('input', 'output'):
        log.debug('Processing stream "%s"', stream_name)
        try:
            stream = getattr(args, stream_name)
        except AttributeError:
            continue
        read_encoding = args.encoding or 'utf-8-sig'
        write_encoding = args.encoding or 'utf-8'
        r_enc = codecs.getreader(read_encoding)
        w_enc = codecs.getwriter(write_encoding)
        log.debug('Got %r as stream', stream)
        if stream in DASH_STREAM_MAP.values():
            log.debug('%s in DASH_STREAM_MAP', stream_name)
            if stream is args.input:
                args.input = srt.parse(r_enc(args.input).read(), ignore_errors=args.ignore_parsing_errors)
            elif stream is args.output:
                args.output = w_enc(args.output)
        else:
            log.debug('%s not in DASH_STREAM_MAP', stream_name)
            if stream is args.input:
                if isinstance(args.input, MutableSequence):
                    for i, input_fn in enumerate(args.input):
                        if input_fn in DASH_STREAM_MAP.values():
                            if stream is args.input:
                                args.input[i] = srt.parse(r_enc(input_fn).read(), ignore_errors=args.ignore_parsing_errors)
                        else:
                            f = r_enc(open(input_fn, 'rb'))
                            with f:
                                args.input[i] = srt.parse(f.read(), ignore_errors=args.ignore_parsing_errors)
                else:
                    f = r_enc(open(stream, 'rb'))
                    with f:
                        args.input = srt.parse(f.read(), ignore_errors=args.ignore_parsing_errors)
            else:
                args.output = w_enc(open(args.output, 'wb'))