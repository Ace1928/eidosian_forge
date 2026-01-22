from decimal import ROUND_FLOOR, Decimal
def quot2(dividend: Decimal, divisor: Decimal) -> Decimal:
    return (dividend / divisor).to_integral_value(rounding=ROUND_FLOOR)