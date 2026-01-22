# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Time series tracking for cache metrics.

This module provides opt-in time series tracking for cache utilization,
hit rate, and token counts with minimal overhead.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache


@dataclass
class NodeStats:
    """Summary statistics about nodes in the radix tree."""

    node_count: int
    leaf_count: int
    avg_node_tokens: float
    max_node_tokens: int
    min_node_tokens: int
    # Histogram buckets: 0-100, 100-500, 500-1000, 1000+
    size_histogram: Dict[str, int]
    # Depth distribution
    depth_distribution: Dict[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "leaf_count": self.leaf_count,
            "avg_node_tokens": round(self.avg_node_tokens, 2),
            "max_node_tokens": self.max_node_tokens,
            "min_node_tokens": self.min_node_tokens,
            "size_histogram": self.size_histogram,
            "depth_distribution": self.depth_distribution,
        }


@dataclass
class CacheSnapshot:
    """A point-in-time snapshot of cache state."""

    timestamp: float
    total_tokens: int
    evictable_tokens: int
    protected_tokens: int
    pool_size: int
    # Token-level tracking
    hit_tokens: int  # Cumulative tokens found in cache
    requested_tokens: int  # Cumulative tokens requested
    request_count: int  # Number of match_prefix calls
    node_stats: Optional[NodeStats] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for API response."""
        utilization = self.total_tokens / self.pool_size if self.pool_size > 0 else 0.0
        miss_tokens = self.requested_tokens - self.hit_tokens
        result = {
            "timestamp": self.timestamp,
            "total_tokens": self.total_tokens,
            "evictable_tokens": self.evictable_tokens,
            "protected_tokens": self.protected_tokens,
            "utilization": utilization,
            "hit_tokens": self.hit_tokens,
            "miss_tokens": miss_tokens,
            "request_count": self.request_count,
        }
        if self.node_stats is not None:
            result["node_stats"] = self.node_stats.to_dict()
        return result


def _collect_node_stats(cache: "RadixCache") -> Optional[NodeStats]:
    """Traverse the radix tree and collect node-level statistics."""
    if cache.disable:
        return None

    node_sizes: List[int] = []
    depth_counts: Dict[int, int] = {}
    leaf_count = 0

    # BFS traversal of the tree
    # Stack contains (node, depth) tuples
    stack = [(cache.root_node, 0)]

    while stack:
        node, depth = stack.pop()

        # Skip root node's empty key
        if node.value is not None and len(node.value) > 0:
            node_size = len(node.value)
            node_sizes.append(node_size)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        # Check if leaf
        if len(node.children) == 0 and node != cache.root_node:
            leaf_count += 1

        # Add children to stack
        for child in node.children.values():
            stack.append((child, depth + 1))

    if not node_sizes:
        return NodeStats(
            node_count=0,
            leaf_count=0,
            avg_node_tokens=0.0,
            max_node_tokens=0,
            min_node_tokens=0,
            size_histogram={"0-100": 0, "100-500": 0, "500-1000": 0, "1000+": 0},
            depth_distribution={},
        )

    # Build histogram
    histogram = {"0-100": 0, "100-500": 0, "500-1000": 0, "1000+": 0}
    for size in node_sizes:
        if size < 100:
            histogram["0-100"] += 1
        elif size < 500:
            histogram["100-500"] += 1
        elif size < 1000:
            histogram["500-1000"] += 1
        else:
            histogram["1000+"] += 1

    return NodeStats(
        node_count=len(node_sizes),
        leaf_count=leaf_count,
        avg_node_tokens=sum(node_sizes) / len(node_sizes),
        max_node_tokens=max(node_sizes),
        min_node_tokens=min(node_sizes),
        size_histogram=histogram,
        depth_distribution=depth_counts,
    )


@dataclass
class CacheTimeSeriesTracker:
    """
    Tracks cache metrics over time using an in-memory circular buffer.

    This class maintains a history of cache snapshots and cumulative hit/miss
    counters to enable time-windowed analysis of cache performance.
    """

    max_history_seconds: float = 300.0  # 5 minutes default
    snapshot_interval: float = 5.0  # 5 seconds between snapshots

    # Internal state
    _snapshots: Deque[CacheSnapshot] = field(default_factory=deque)
    _total_requests: int = 0
    _hit_tokens: int = 0
    _requested_tokens: int = 0
    _last_snapshot_time: float = 0.0

    def __post_init__(self):
        self._snapshots = deque()
        self._total_requests = 0
        self._hit_tokens = 0
        self._requested_tokens = 0
        self._last_snapshot_time = 0.0

    def record_match(self, matched_tokens: int, requested_tokens: int) -> None:
        """Record a prefix match with token-level granularity.

        Args:
            matched_tokens: Number of tokens that were found in cache.
            requested_tokens: Total number of tokens requested.
        """
        self._total_requests += 1
        self._hit_tokens += matched_tokens
        self._requested_tokens += requested_tokens

    def take_snapshot(self, cache: "RadixCache") -> Optional[CacheSnapshot]:
        """
        Capture current cache state and add to history.

        Prunes old entries beyond max_history_seconds. Returns the new
        snapshot if one was taken, None if called too soon after last snapshot.
        """
        now = time.time()

        # Rate limit snapshots
        if now - self._last_snapshot_time < self.snapshot_interval:
            return None

        self._last_snapshot_time = now

        # Get pool size (total capacity in tokens)
        pool_size = 0
        if cache.token_to_kv_pool_allocator is not None:
            pool_size = cache.token_to_kv_pool_allocator.size

        # Collect node-level statistics
        node_stats = _collect_node_stats(cache)

        # Create snapshot
        snapshot = CacheSnapshot(
            timestamp=now,
            total_tokens=cache.total_size(),
            evictable_tokens=cache.evictable_size(),
            protected_tokens=cache.protected_size(),
            pool_size=pool_size,
            hit_tokens=self._hit_tokens,
            requested_tokens=self._requested_tokens,
            request_count=self._total_requests,
            node_stats=node_stats,
        )

        self._snapshots.append(snapshot)

        # Prune old entries
        cutoff = now - self.max_history_seconds
        while self._snapshots and self._snapshots[0].timestamp < cutoff:
            self._snapshots.popleft()

        return snapshot

    def get_stats_dict(self) -> Dict[str, Any]:
        """Return summary statistics for API response."""
        latest = self._snapshots[-1] if self._snapshots else None

        total_tokens = latest.total_tokens if latest else 0
        evictable_tokens = latest.evictable_tokens if latest else 0
        protected_tokens = latest.protected_tokens if latest else 0
        pool_size = latest.pool_size if latest else 0
        utilization = total_tokens / pool_size if pool_size > 0 else 0.0

        result = {
            "total_tokens": total_tokens,
            "evictable_tokens": evictable_tokens,
            "protected_tokens": protected_tokens,
            "utilization": utilization,
            "total_hit_tokens": self._hit_tokens,
            "total_miss_tokens": self._requested_tokens - self._hit_tokens,
            "total_requests": self._total_requests,
        }

        # Include node stats if available
        if latest and latest.node_stats is not None:
            result["node_stats"] = latest.node_stats.to_dict()

        return result

    def get_history(
        self, window_seconds: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Return time series data for graphing.

        Args:
            window_seconds: Time window in seconds. If None, returns all history.

        Returns:
            List of snapshot dictionaries ordered by timestamp.
        """
        if window_seconds is None:
            return [s.to_dict() for s in self._snapshots]

        now = time.time()
        cutoff = now - window_seconds
        return [s.to_dict() for s in self._snapshots if s.timestamp >= cutoff]
