def interrupt_main():
    """Set _interrupt flag to True to have start_new_thread raise
    KeyboardInterrupt upon exiting."""
    if _main:
        raise KeyboardInterrupt
    else:
        global _interrupt
        _interrupt = True