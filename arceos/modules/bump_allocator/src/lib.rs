#![no_std]
use allocator::{AllocError, BaseAllocator, ByteAllocator, PageAllocator};
use core::sync::atomic::{AtomicUsize, Ordering};

// Early memory allocator
// Use it before formal bytes-allocator and pages-allocator can work!
/// This is a double-end memory range:
/// - Alloc bytes forward
/// - Alloc pages backward
///
/// [ bytes-used | avail-area | pages-used ]
/// |            | -->  <-- |            |
/// start     b_pos      p_pos          end
///
/// For bytes area, count records number of allocations.
/// When it goes down to ZERO, free bytes-used area.
/// For pages area, it will never be freed!
static START: AtomicUsize = AtomicUsize::new(0);
static END: AtomicUsize = AtomicUsize::new(0);
static B_POS: AtomicUsize = AtomicUsize::new(0);
static P_POS: AtomicUsize = AtomicUsize::new(0);
static COUNT: AtomicUsize = AtomicUsize::new(0);

const fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

const fn align_down(value: usize, align: usize) -> usize {
    value & !(align - 1)
}

pub struct EarlyAllocator<const SIZE: usize> {}

impl<const SIZE: usize> EarlyAllocator<SIZE> {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<const SIZE: usize> BaseAllocator for EarlyAllocator<SIZE> {
    fn init(&mut self, start: usize, size: usize) {
        START.store(start, Ordering::Relaxed);
        END.store(start + size, Ordering::Relaxed);
        B_POS.store(start, Ordering::Relaxed);
        P_POS.store(start + size, Ordering::Relaxed);
        COUNT.store(0, Ordering::Relaxed);
    }

    fn add_memory(&mut self, start: usize, size: usize) -> allocator::AllocResult {
        // Only a region following the current one can extend this single-range
        // bump allocator. A disjoint region is ignored so that boot can
        // continue; the largest region (set by init) is the heap.
        if start == END.load(Ordering::Relaxed) {
            END.store(start + size, Ordering::Relaxed);
            P_POS.store(start + size, Ordering::Relaxed);
        }
        Ok(())
    }
}

impl<const SIZE: usize> ByteAllocator for EarlyAllocator<SIZE> {
    fn alloc(
        &mut self,
        layout: core::alloc::Layout,
    ) -> allocator::AllocResult<core::ptr::NonNull<u8>> {
        let size = layout.size();
        let align = layout.align();
        let b_pos = B_POS.load(Ordering::Relaxed);
        let p_pos = P_POS.load(Ordering::Relaxed);
        if size == 0 {
            // Zero-sized allocation: hand out an aligned dangling pointer.
            let ptr = align_up(b_pos, align);
            return core::ptr::NonNull::new(ptr as *mut u8).ok_or(AllocError::NoMemory);
        }
        let pos = align_up(b_pos, align);
        let new_b_pos = pos.checked_add(size).ok_or(AllocError::NoMemory)?;
        if new_b_pos > p_pos {
            return Err(AllocError::NoMemory);
        }
        B_POS.store(new_b_pos, Ordering::Relaxed);
        COUNT.fetch_add(1, Ordering::Relaxed);
        core::ptr::NonNull::new(pos as *mut u8).ok_or(AllocError::NoMemory)
    }

    fn dealloc(&mut self, _pos: core::ptr::NonNull<u8>, layout: core::alloc::Layout) {
        if layout.size() == 0 {
            return;
        }
        if COUNT.load(Ordering::Relaxed) == 0 {
            return;
        }
        if COUNT.fetch_sub(1, Ordering::Relaxed) == 1 {
            // The last live byte allocation was freed: rewind the bump pointer.
            B_POS.store(START.load(Ordering::Relaxed), Ordering::Relaxed);
        }
    }

    fn total_bytes(&self) -> usize {
        END.load(Ordering::Relaxed) - START.load(Ordering::Relaxed)
    }

    fn used_bytes(&self) -> usize {
        B_POS.load(Ordering::Relaxed) - START.load(Ordering::Relaxed)
    }

    fn available_bytes(&self) -> usize {
        P_POS.load(Ordering::Relaxed).saturating_sub(B_POS.load(Ordering::Relaxed))
    }
}

impl<const SIZE: usize> PageAllocator for EarlyAllocator<SIZE> {
    const PAGE_SIZE: usize = SIZE;

    fn alloc_pages(
        &mut self,
        num_pages: usize,
        align_pow2: usize,
    ) -> allocator::AllocResult<usize> {
        if END.load(Ordering::Relaxed) == 0 {
            return Err(AllocError::NoMemory);
        }
        let align = align_pow2.max(1);
        let size = num_pages.checked_mul(SIZE).ok_or(AllocError::NoMemory)?;
        let b_pos = B_POS.load(Ordering::Relaxed);
        let p_pos = P_POS.load(Ordering::Relaxed);
        let new_p_pos = align_down(p_pos.saturating_sub(size), align);
        if new_p_pos < b_pos {
            return Err(AllocError::NoMemory);
        }
        P_POS.store(new_p_pos, Ordering::Relaxed);
        Ok(new_p_pos)
    }

    fn dealloc_pages(&mut self, _pos: usize, _num_pages: usize) {}

    fn total_pages(&self) -> usize {
        self.total_bytes() / SIZE
    }

    fn used_pages(&self) -> usize {
        (END.load(Ordering::Relaxed) - P_POS.load(Ordering::Relaxed)) / SIZE
    }

    fn available_pages(&self) -> usize {
        self.available_bytes() / SIZE
    }
}
