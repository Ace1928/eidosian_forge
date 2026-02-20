"""Tests for word_forge.queue.queue_config module.

This module tests the QueueConfig dataclass and related enums/functions
for queue processing configuration.
"""

from word_forge.queue.queue_config import (
    QueueConfig,
    QueueMetricsFormat,
    QueuePerformanceProfile,
)


class TestQueueConfig:
    """Tests for the QueueConfig dataclass."""

    def test_default_values(self):
        """Test that QueueConfig has sensible defaults."""
        config = QueueConfig()
        assert config.batch_size == 50
        assert config.throttle_seconds == 0.1
        assert config.lru_cache_size == 128
        assert config.max_queue_size is None
        assert config.apply_default_normalization is True
        assert config.use_threading is True
        assert config.lock_type == "reentrant"
        assert config.track_metrics is False
        assert config.metrics_format == QueueMetricsFormat.JSON
        assert config.max_sample_size == 100
        assert config.performance_profile is None

    def test_custom_values(self):
        """Test QueueConfig with custom values."""
        config = QueueConfig(
            batch_size=100,
            throttle_seconds=0.5,
            lru_cache_size=256,
            max_queue_size=1000,
            apply_default_normalization=False,
            use_threading=False,
            lock_type="standard",
            track_metrics=True,
            metrics_format=QueueMetricsFormat.PROMETHEUS,
            max_sample_size=50,
            performance_profile=QueuePerformanceProfile.HIGH_THROUGHPUT,
        )
        assert config.batch_size == 100
        assert config.throttle_seconds == 0.5
        assert config.lru_cache_size == 256
        assert config.max_queue_size == 1000
        assert config.apply_default_normalization is False
        assert config.use_threading is False
        assert config.lock_type == "standard"
        assert config.track_metrics is True
        assert config.metrics_format == QueueMetricsFormat.PROMETHEUS
        assert config.max_sample_size == 50
        assert config.performance_profile == QueuePerformanceProfile.HIGH_THROUGHPUT


class TestEffectiveThroughput:
    """Tests for the effective_throughput cached property."""

    def test_default_throughput(self):
        """Test effective throughput with default values."""
        config = QueueConfig()
        # 50 / 0.1 = 500 items per second
        assert config.effective_throughput == 500.0

    def test_custom_throughput(self):
        """Test effective throughput with custom values."""
        config = QueueConfig(batch_size=100, throttle_seconds=0.5)
        # 100 / 0.5 = 200 items per second
        assert config.effective_throughput == 200.0

    def test_zero_throttle(self):
        """Test effective throughput with zero throttle."""
        config = QueueConfig(batch_size=50, throttle_seconds=0)
        # Should return batch_size when throttle is 0
        assert config.effective_throughput == 50.0

    def test_negative_throttle(self):
        """Test effective throughput with negative throttle."""
        config = QueueConfig(batch_size=50, throttle_seconds=-0.1)
        # Should return batch_size for negative throttle
        assert config.effective_throughput == 50.0


class TestWithBatchSize:
    """Tests for the with_batch_size method."""

    def test_returns_new_instance(self):
        """Test that with_batch_size returns a new instance."""
        config1 = QueueConfig()
        config2 = config1.with_batch_size(100)
        assert config1 is not config2

    def test_updates_batch_size(self):
        """Test that batch_size is updated."""
        config1 = QueueConfig(batch_size=50)
        config2 = config1.with_batch_size(100)
        assert config2.batch_size == 100
        assert config1.batch_size == 50  # Original unchanged

    def test_preserves_other_values(self):
        """Test that other values are preserved."""
        config1 = QueueConfig(
            batch_size=50,
            throttle_seconds=0.2,
            lru_cache_size=256,
            track_metrics=True,
        )
        config2 = config1.with_batch_size(100)
        assert config2.throttle_seconds == 0.2
        assert config2.lru_cache_size == 256
        assert config2.track_metrics is True


class TestWithPerformanceProfile:
    """Tests for the with_performance_profile method."""

    def test_balanced_profile(self):
        """Test BALANCED performance profile."""
        config = QueueConfig().with_performance_profile(QueuePerformanceProfile.BALANCED)
        assert config.performance_profile == QueuePerformanceProfile.BALANCED
        # BALANCED uses defaults
        assert config.batch_size == 50
        assert config.throttle_seconds == 0.1
        assert config.lru_cache_size == 128

    def test_low_latency_profile(self):
        """Test LOW_LATENCY performance profile."""
        config = QueueConfig().with_performance_profile(QueuePerformanceProfile.LOW_LATENCY)
        assert config.performance_profile == QueuePerformanceProfile.LOW_LATENCY
        assert config.batch_size == 10
        assert config.throttle_seconds == 0.01
        assert config.lru_cache_size == 256

    def test_high_throughput_profile(self):
        """Test HIGH_THROUGHPUT performance profile."""
        config = QueueConfig().with_performance_profile(QueuePerformanceProfile.HIGH_THROUGHPUT)
        assert config.performance_profile == QueuePerformanceProfile.HIGH_THROUGHPUT
        assert config.batch_size == 200
        assert config.throttle_seconds == 0.5
        assert config.lru_cache_size == 512

    def test_memory_efficient_profile(self):
        """Test MEMORY_EFFICIENT performance profile."""
        config = QueueConfig().with_performance_profile(QueuePerformanceProfile.MEMORY_EFFICIENT)
        assert config.performance_profile == QueuePerformanceProfile.MEMORY_EFFICIENT
        assert config.batch_size == 25
        assert config.throttle_seconds == 0.3
        assert config.lru_cache_size == 64

    def test_none_profile_resets_to_defaults(self):
        """Test that None profile resets to defaults."""
        config = QueueConfig(batch_size=1000).with_performance_profile(None)
        assert config.performance_profile is None
        # Should reset to defaults
        assert config.batch_size == 50

    def test_preserves_non_profile_settings(self):
        """Test that non-profile settings are preserved."""
        config = QueueConfig(
            max_queue_size=500,
            use_threading=False,
            track_metrics=True,
        ).with_performance_profile(QueuePerformanceProfile.HIGH_THROUGHPUT)
        assert config.max_queue_size == 500
        assert config.use_threading is False
        assert config.track_metrics is True


class TestOptimizeForThreading:
    """Tests for the optimize_for_threading method."""

    def test_enable_threading(self):
        """Test enabling threading optimizations."""
        config = QueueConfig(use_threading=False, lock_type="standard").optimize_for_threading(True)
        assert config.use_threading is True
        assert config.lock_type == "reentrant"

    def test_disable_threading(self):
        """Test disabling threading optimizations."""
        config = QueueConfig(batch_size=50, use_threading=True).optimize_for_threading(False)
        assert config.use_threading is False
        # Batch size should increase (up to 500)
        assert config.batch_size == 100  # 50 * 2

    def test_disable_threading_respects_max_batch(self):
        """Test that batch size doesn't exceed 500 when disabling threading."""
        config = QueueConfig(batch_size=300).optimize_for_threading(False)
        assert config.batch_size == 500  # Capped at 500

    def test_preserves_other_settings(self):
        """Test that other settings are preserved."""
        config = QueueConfig(
            throttle_seconds=0.5,
            track_metrics=True,
            max_queue_size=1000,
        ).optimize_for_threading(True)
        assert config.throttle_seconds == 0.5
        assert config.track_metrics is True
        assert config.max_queue_size == 1000


class TestGetMetricsConfig:
    """Tests for the get_metrics_config method."""

    def test_metrics_disabled(self):
        """Test metrics config when metrics are disabled."""
        config = QueueConfig(track_metrics=False)
        metrics_config = config.get_metrics_config()
        assert metrics_config["enabled"] is False
        assert metrics_config["format"] == QueueMetricsFormat.JSON
        assert metrics_config["max_sample_size"] == 100

    def test_metrics_enabled(self):
        """Test metrics config when metrics are enabled."""
        config = QueueConfig(
            track_metrics=True,
            metrics_format=QueueMetricsFormat.PROMETHEUS,
            max_sample_size=50,
        )
        metrics_config = config.get_metrics_config()
        assert metrics_config["enabled"] is True
        assert metrics_config["format"] == QueueMetricsFormat.PROMETHEUS
        assert metrics_config["max_sample_size"] == 50


class TestQueueMetricsFormat:
    """Tests for the QueueMetricsFormat enum."""

    def test_json_format(self):
        """Test JSON format value."""
        assert QueueMetricsFormat.JSON.value == "json"

    def test_prometheus_format(self):
        """Test PROMETHEUS format value."""
        assert QueueMetricsFormat.PROMETHEUS.value == "prometheus"

    def test_is_string_enum(self):
        """Test that format is string enum."""
        assert isinstance(QueueMetricsFormat.JSON, str)
        assert isinstance(QueueMetricsFormat.PROMETHEUS, str)


class TestQueuePerformanceProfile:
    """Tests for the QueuePerformanceProfile enum."""

    def test_all_profiles_exist(self):
        """Test that all expected profiles exist."""
        assert QueuePerformanceProfile.BALANCED
        assert QueuePerformanceProfile.LOW_LATENCY
        assert QueuePerformanceProfile.HIGH_THROUGHPUT
        assert QueuePerformanceProfile.MEMORY_EFFICIENT

    def test_profile_count(self):
        """Test that there are 4 profiles."""
        assert len(QueuePerformanceProfile) == 4


class TestEnvVars:
    """Tests for environment variable mapping."""

    def test_env_vars_defined(self):
        """Test that ENV_VARS is defined."""
        assert hasattr(QueueConfig, "ENV_VARS")
        assert isinstance(QueueConfig.ENV_VARS, dict)

    def test_env_vars_contains_batch_size(self):
        """Test that batch size env var is defined."""
        assert "WORD_FORGE_QUEUE_BATCH_SIZE" in QueueConfig.ENV_VARS

    def test_env_vars_contains_throttle(self):
        """Test that throttle env var is defined."""
        assert "WORD_FORGE_QUEUE_THROTTLE" in QueueConfig.ENV_VARS

    def test_env_vars_contains_cache_size(self):
        """Test that cache size env var is defined."""
        assert "WORD_FORGE_QUEUE_CACHE_SIZE" in QueueConfig.ENV_VARS

    def test_env_vars_contains_max_size(self):
        """Test that max queue size env var is defined."""
        assert "WORD_FORGE_QUEUE_MAX_SIZE" in QueueConfig.ENV_VARS

    def test_env_vars_contains_use_threads(self):
        """Test that use threads env var is defined."""
        assert "WORD_FORGE_QUEUE_USE_THREADS" in QueueConfig.ENV_VARS

    def test_env_vars_contains_lock_type(self):
        """Test that lock type env var is defined."""
        assert "WORD_FORGE_QUEUE_LOCK_TYPE" in QueueConfig.ENV_VARS

    def test_env_vars_contains_track_metrics(self):
        """Test that track metrics env var is defined."""
        assert "WORD_FORGE_QUEUE_TRACK_METRICS" in QueueConfig.ENV_VARS

    def test_env_vars_contains_metrics_format(self):
        """Test that metrics format env var is defined."""
        assert "WORD_FORGE_QUEUE_METRICS_FORMAT" in QueueConfig.ENV_VARS

    def test_env_vars_contains_performance_profile(self):
        """Test that performance profile env var is defined."""
        assert "WORD_FORGE_QUEUE_PERFORMANCE_PROFILE" in QueueConfig.ENV_VARS
