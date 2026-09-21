# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import weakref
from typing import ClassVar, Optional

import torch
import torch.distributed as dist
from compressed_tensors.distributed import is_source_process
from compressed_tensors.offload.cache.disk import DiskCache
from compressed_tensors.offload.cache.dist_batch import BatchedOffloadMixin
from compressed_tensors.offload.utils import send_tensors, to_tensor


class DistributedDiskCache(BatchedOffloadMixin, DiskCache):
    """
    Handles offloading and onloading tensors from/to disk. For more information, see
    `compressed_tensors.offload.cache.disk_cache::DiskCache`.
    """

    _defer_file_deletion_depth = 0
    _deferred_file_deletions: set[str] = set()
    # MutableMapping instances are unhashable, so use weak values keyed by id.
    _instances: ClassVar[
        weakref.WeakValueDictionary[int, "DistributedDiskCache"]
    ] = weakref.WeakValueDictionary()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instances[id(self)] = self

    @classmethod
    @contextlib.contextmanager
    def defer_file_deletions(cls):
        """Delay shared-file deletion until distributed workers synchronize."""
        cls._defer_file_deletion_depth += 1
        try:
            yield
        finally:
            cls._defer_file_deletion_depth -= 1

    @classmethod
    def flush_deferred_file_deletions(cls) -> None:
        """Delete queued files that no distributed rank still references."""
        pending = set(cls._deferred_file_deletions)
        live_paths = set()
        for cache in cls._instances.values():
            for offloaded in cache.offloaded_values.values():
                weight_info = cache.index.get(offloaded)
                if weight_info is not None:
                    file_path = weight_info["safetensors_file"]
                    if file_path in pending:
                        live_paths.add(file_path)

        if dist.is_available() and dist.is_initialized():
            gathered_live_paths = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_live_paths, live_paths)
            protected_paths = set().union(
                *(paths for paths in gathered_live_paths if paths is not None)
            )
        else:
            protected_paths = live_paths

        for file_path in pending - protected_paths:
            if os.path.lexists(file_path):
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    # Another rank may have reclaimed the same stale path first.
                    pass

        # Retry protected files during a later round after references disappear.
        cls._deferred_file_deletions.intersection_update(protected_paths)

    def _remove_file(self, file_path: str) -> None:
        if self._defer_file_deletion_depth > 0:
            self._deferred_file_deletions.add(file_path)
        else:
            super()._remove_file(file_path)

    def offload_local(self, tensor: torch.Tensor) -> tuple[torch.Tensor, list]:
        """
        Write tensor data to disk on the source rank.

        :param tensor: tensor on any device
        :return: meta tensor representing the disk offloaded tensor, and the file
            location, dtype and shape used by other ranks to rebuild it
        """
        offloaded = DiskCache.offload(self, tensor)
        weight_info = self.index[offloaded]
        return offloaded, [
            weight_info["safetensors_file"],
            weight_info["weight_name"],
            weight_info["dtype"],
            offloaded.shape,
        ]

    def recv_offload(
        self, tensor: torch.Tensor, metadata: list, memo: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Point a meta tensor at the file written by the source rank.

        :param tensor: this rank's local tensor (often a meta tensor)
        :param metadata: file location, dtype and shape from the source rank
        :param memo: unused, accepted for interface compatibility
        :return: meta tensor representing the disk offloaded tensor
        """
        safetensors_file, weight_name, dtype, src_shape = metadata
        offloaded = send_tensors(tensor, device="meta")
        src_dtype = getattr(torch, dtype)

        # transformers may init params/buffers on non-source (meta) ranks with a
        # different dtype or shape than the checkpoint (e.g. `inv_freq`, or
        # tied/multimodal weights), so rebuild the meta tensor to match the source.
        if offloaded.dtype != src_dtype or offloaded.shape != src_shape:
            empty = torch.empty(src_shape, dtype=src_dtype, device="meta")
            offloaded = to_tensor(empty, offloaded)

        self.index[offloaded] = {
            "safetensors_file": safetensors_file,
            "weight_name": weight_name,
            "dtype": dtype,
        }
        return offloaded

    def __delitem__(self, key: str):
        """
        Remove the offload associated with `key`. If a new file was created to store
        updated tensor data, that new tensor data file is deleted.

        Any references to onloaded tensors held by this class are invalidated.

        :param key: name of tensor to invalidate
        """
        if is_source_process():
            super().__delitem__(key)
        else:
            if not self.onloading_disabled:
                offloaded = self.offloaded_values[key]
                del self.index[offloaded]
            super(DiskCache, self).__delitem__(key)
