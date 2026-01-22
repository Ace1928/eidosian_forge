from os_brick.encryptors import base
A VolumeEncryptor that does nothing.

    This class exists solely to wrap regular (i.e., unencrypted) volumes so
    that they do not require special handling with respect to an encrypted
    volume. This implementation performs no action when a volume is attached
    or detached.
    