import time
import rsa
def run_speed_test(bitsize):
    iterations = 0
    start = end = time.time()
    while iterations < 10 or end - start < 2:
        iterations += 1
        rsa.newkeys(bitsize, accurate=accurate, poolsize=poolsize)
        end = time.time()
    duration = end - start
    dur_per_call = duration / iterations
    print('%5i bit: %9.3f sec. (%i iterations over %.1f seconds)' % (bitsize, dur_per_call, iterations, duration))