import os
def has_chief_oracle():
    """Checks for distributed tuning with a chief Oracle.

    `CloudOracle` manages its own distribution so should not set
    "KERASTUNER_ORACLE_IP".

    Returns:
        Boolean, whether distributed tuning with a chief Oracle should be run.
    """
    if 'KERASTUNER_ORACLE_IP' in os.environ:
        if 'KERASTUNER_ORACLE_PORT' not in os.environ:
            raise RuntimeError('Environment variable "KERASTUNER_ORACLE_IP" was set, but "KERASTUNER_ORACLE_PORT" was not. Please specify a port.')
        if 'KERASTUNER_TUNER_ID' not in os.environ:
            raise RuntimeError('Environment variable "KERASTUNER_ORACLE_IP" was set, but "KERASTUNER_TUNER_ID" was not. Please specify an ID for each tuner.')
        return True
    return False