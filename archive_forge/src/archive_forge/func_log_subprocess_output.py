import logging
def log_subprocess_output(output):
    if output:
        for line in output.rstrip().splitlines():
            converter_logger.debug('subprocess output: %s', line.rstrip())