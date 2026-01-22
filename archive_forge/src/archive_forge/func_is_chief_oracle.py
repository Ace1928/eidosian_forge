import os
def is_chief_oracle():
    if has_chief_oracle():
        return 'chief' in os.environ['KERASTUNER_TUNER_ID']
    return False