import re
def number_to_words(self, num: int) -> str:
    """
        Converts numbers(`int`) to words(`str`).

        Please note that it only supports upto - "'nine hundred ninety-nine quadrillion, nine hundred ninety-nine
        trillion, nine hundred ninety-nine billion, nine hundred ninety-nine million, nine hundred ninety-nine
        thousand, nine hundred ninety-nine'" or `number_to_words(999_999_999_999_999_999)`.
        """
    if num == 0:
        return 'zero'
    elif num < 0:
        return 'minus ' + self.number_to_words(abs(num))
    elif num < 10:
        return self.ones[num]
    elif num < 20:
        return self.teens[num - 10]
    elif num < 100:
        return self.tens[num // 10] + ('-' + self.number_to_words(num % 10) if num % 10 != 0 else '')
    elif num < 1000:
        return self.ones[num // 100] + ' hundred' + (' ' + self.number_to_words(num % 100) if num % 100 != 0 else '')
    elif num < 1000000:
        return self.number_to_words(num // 1000) + ' thousand' + (', ' + self.number_to_words(num % 1000) if num % 1000 != 0 else '')
    elif num < 1000000000:
        return self.number_to_words(num // 1000000) + ' million' + (', ' + self.number_to_words(num % 1000000) if num % 1000000 != 0 else '')
    elif num < 1000000000000:
        return self.number_to_words(num // 1000000000) + ' billion' + (', ' + self.number_to_words(num % 1000000000) if num % 1000000000 != 0 else '')
    elif num < 1000000000000000:
        return self.number_to_words(num // 1000000000000) + ' trillion' + (', ' + self.number_to_words(num % 1000000000000) if num % 1000000000000 != 0 else '')
    elif num < 1000000000000000000:
        return self.number_to_words(num // 1000000000000000) + ' quadrillion' + (', ' + self.number_to_words(num % 1000000000000000) if num % 1000000000000000 != 0 else '')
    else:
        return 'number out of range'