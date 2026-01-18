"""
Demonstration of QueueManager functionality.
"""

from word_forge.queue.queue_manager import QueueManager, TaskPriority


def main() -> None:
    """
    Demonstrate QueueManager functionality with basic operations.

    This function provides a simple demonstration of key queue operations:
    - Creating a queue manager
    - Enqueueing items with various priorities
    - Dequeuing items
    - Monitoring queue metrics

    Examples:
        >>> # Run the demonstration
        >>> main()
    """
    # Create a string queue manager
    queue_manager: QueueManager[str] = QueueManager(size_limit=100)

    print("QueueManager Demo")
    print("-----------------")

    # Add items with different priorities
    print("\nAdding items with different priorities...")
    queue_manager.enqueue("high-priority item", TaskPriority.HIGH)
    queue_manager.enqueue("normal item 1", TaskPriority.NORMAL)
    queue_manager.enqueue("normal item 2", TaskPriority.NORMAL)
    queue_manager.enqueue("low-priority item", TaskPriority.LOW)

    # Try to add a duplicate
    result = queue_manager.enqueue("normal item 1")
    print(f"Adding duplicate item: {'Rejected' if not result.unwrap() else 'Added'}")

    # Show queue stats
    print(f"\nQueue size: {queue_manager.size}")
    print(f"Items enqueued: {queue_manager.metrics.enqueued_count}")
    print(f"Duplicates rejected: {queue_manager.metrics.rejected_count}")

    # Dequeue and process items
    print("\nProcessing items in priority order:")
    while not queue_manager.is_empty:
        result = queue_manager.dequeue()
        if result.is_success:
            item = result.unwrap()
            print(f"- Processing: {item}")

    # Show final stats
    print("\nFinal statistics:")
    stats = queue_manager.get_stats()
    print(f"Items processed: {stats['dequeued_count']}")
    print(f"High priority: {stats['high_priority_count']}")
    print(f"Normal priority: {stats['normal_priority_count']}")
    print(f"Low priority: {stats['low_priority_count']}")

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
