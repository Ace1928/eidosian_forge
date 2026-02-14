__kernel void parallelGreedy2(__global long* denominators, int numDenominators, double numberToExchange, __global char* result) {
    for (int i = 0; i < numDenominators - 1; i++) {
        for (int j = i + 1; j < numDenominators; j++) {
            if (denominators[i] < denominators[j]) {
                long temp = denominators[i];
                denominators[i] = denominators[j];
                denominators[j] = temp;
            }
        }
    }

    result[0] = '\0';

    for (int i = 0; i < numDenominators && numberToExchange > 0; i++) {
        if (numberToExchange >= denominators[i]) {
            int biggestNumberToExchangeInLoop = (int)(numberToExchange / denominators[i]);
            char temp[100];
            snprintf(temp, sizeof(temp), "%ld cash x%d\n", denominators[i], biggestNumberToExchangeInLoop);

            int offset = 0;
            while (result[offset] != '\0') {
                offset++;
            }

            for (int j = 0; temp[j] != '\0'; j++) {
                result[offset++] = temp[j];
            }
            result[offset] = '\0';

            numberToExchange = round(100 * (numberToExchange - (biggestNumberToExchangeInLoop * denominators[i]))) / 100.0;
        }
    }
}
