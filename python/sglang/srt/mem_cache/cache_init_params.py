from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


@dataclasses.dataclass
class CacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int

    is_eagle: bool = False
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    eviction_policy: str = "lru"
    disable_finished_insert: bool = False

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False

    # Time series cache metrics tracking
    enable_cache_timeseries: bool = False
    cache_timeseries_interval: float = 5.0
    cache_timeseries_history: float = 300.0

    enable_mamba_extra_buffer: bool = False

    # For SWAChunkCache
    sliding_window_size: Optional[int] = None
    attention_chunk_size: Optional[int] = None

    pp_rank: int = 0
    pp_size: int = 1

    chunked_prefill_size: Optional[int] = None
