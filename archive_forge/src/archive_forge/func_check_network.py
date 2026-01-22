import platform, subprocess, sys, os
import socket, time
import argparse
def check_network(args):
    print('----------Network Test----------')
    if args.timeout > 0:
        print('Setting timeout: {}'.format(args.timeout))
        socket.setdefaulttimeout(10)
    for region in args.region.strip().split(','):
        r = region.strip().lower()
        if not r:
            continue
        if r in REGIONAL_URLS:
            URLS.update(REGIONAL_URLS[r])
        else:
            import warnings
            warnings.warn('Region {} do not need specific test, please refer to global sites.'.format(r))
    for name, url in URLS.items():
        test_connection(name, url, args.timeout)