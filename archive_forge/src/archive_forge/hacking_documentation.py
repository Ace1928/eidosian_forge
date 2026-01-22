import sys
import fixtures
Fixtures for checking translation rules.

    1. Exception messages should be translated
    2. Logging messages should not be translated
    3. If a message is used for both an exception and logging it
       should be translated
    