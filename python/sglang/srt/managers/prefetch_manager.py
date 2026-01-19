import torch

class PrefetchManager:
    def __init__(self, mem_manager, radix_cache):
        self.mem = mem_manager          # Handles memory allocation (alloc_gpu)
        self.tree = radix_cache         # The data structure
        self.stream = torch.cuda.Stream() # The dedicated transfer lane

    def tick(self):
        """Call this once per scheduling loop to update statuses."""
        # 1. Check for completed transfers
        loading_nodes = [n for n in self.tree.get_all_nodes() if n.is_prefetching]
        
        for node in loading_nodes:
            if node.prefetch_event.query(): # Returns True if copy is done
                # Mark as fully loaded
                node.is_prefetching = False
                node.value = node.pending_gpu_indices # Commit the new indices
                node.pending_gpu_indices = None
                print(f"Node {node.id} is now hot in GPU!")

    def request_prefetch(self, node):
        """The main method you call when you predict a node is needed."""
        
        # Validation checks (Simplifying the logic)
        if node.value is not None: return # Already in GPU
        if node.host_value is None: return # Not in CPU (can't fetch)
        if node.is_prefetching: return    # Already busy
        
        # 1. Try to allocate space on GPU
        # (This uses SGLang's existing allocator)
        needed_slots = len(node.host_value)
        gpu_indices = self.mem.alloc_gpu(needed_slots)
        
        if gpu_indices is None:
            return # GPU is full, abort safely

        # 2. Start the Transfer
        node.is_prefetching = True
        node.pending_gpu_indices = gpu_indices # Store temporarily
        
        with torch.cuda.stream(self.stream):
            # The Magic Move: Copy CPU indices to GPU indices
            # (Requires access to the actual big tensor buffers)
            self.mem.copy_from_cpu(
                src_indices=node.host_value, 
                dst_indices=gpu_indices
            )

        # 3. Set the Flag
        event = torch.cuda.Event()
        event.record(self.stream)
        node.prefetch_event = event