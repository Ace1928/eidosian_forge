import cProfile
import pstats

def profile_main():
    """
    Profile the main function to identify bottlenecks.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        main()
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumulative')
        stats.print_stats()

if __name__ == "__main__":
    profile_main()
