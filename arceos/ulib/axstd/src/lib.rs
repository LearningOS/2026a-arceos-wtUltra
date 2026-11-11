//! # The ArceOS Standard Library
//!
//! The [ArceOS] Standard Library is a mini-std library, with an interface similar
//! to rust [std], but calling the functions directly in ArceOS modules, instead
//! of using libc and system calls.
//!
//! These features are exactly the same as those in [axfeat], they are used to
//! provide users with the selection of features in axfeat, without import
//! [axfeat] additionally:
//!
//! ## Cargo Features
//!
//! - CPU
//!     - `smp`: Enable SMP (symmetric multiprocessing) support.
//!     - `fp_simd`: Enable floating point and SIMD support.
//! - Interrupts:
//!     - `irq`: Enable interrupt handling support.
//! - Memory
//!     - `alloc`: Enable dynamic memory allocation.
//!     - `alloc-tlsf`: Use the TLSF allocator.
//!     - `alloc-slab`: Use the slab allocator.
//!     - `alloc-buddy`: Use the buddy system allocator.
//!     - `paging`: Enable page table manipulation.
//!     - `tls`: Enable thread-local storage.
//! - Task management
//!     - `multitask`: Enable multi-threading support.
//!     - `sched_fifo`: Use the FIFO cooperative scheduler.
//!     - `sched_rr`: Use the Round-robin preemptive scheduler.
//!     - `sched_cfs`: Use the Completely Fair Scheduler (CFS) preemptive scheduler.
//! - Upperlayer stacks
//!     - `fs`: Enable file system support.
//!     - `myfs`: Allow users to define their custom filesystems to override the default.
//!     - `net`: Enable networking support.
//!     - `dns`: Enable DNS lookup support.
//!     - `display`: Enable graphics support.
//! - Device drivers
//!     - `bus-mmio`: Use device tree to probe all MMIO devices.
//!     - `bus-pci`: Use PCI bus to probe all PCI devices.
//!     - `driver-ramdisk`: Use the RAM disk to emulate the block device.
//!     - `driver-ixgbe`: Enable the Intel 82599 10Gbit NIC driver.
//!     - `driver-bcm2835-sdhci`: Enable the BCM2835 SDHCI driver (Raspberry Pi SD card).
//! - Logging
//!     - `log-level-off`: Disable all logging.
//!     - `log-level-error`, `log-level-warn`, `log-level-info`, `log-level-debug`,
//!       `log-level-trace`: Keep logging only at the specified level or higher.
//!
//! [ArceOS]: https://github.com/arceos-org/arceos

#![cfg_attr(all(not(test), not(doc)), no_std)]
#![feature(doc_cfg)]
#![feature(doc_auto_cfg)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
#[doc(no_inline)]
pub use alloc::{boxed, format, string, vec};

/// The alloc collections, with a HashMap type added.
/// The alloc crate does not provide a hash map, so HashMap is provided
/// here as a real hash table (open addressing with FNV-1a hashing, linear
/// probing and tombstones), which lets kernel apps use
/// std::collections::HashMap.
#[cfg(feature = "alloc")]
pub mod collections {
    pub use alloc::collections::*;

    use alloc::vec::{IntoIter as VecIntoIter, Vec};
    use core::borrow::Borrow;
    use core::fmt;
    use core::hash::{Hash, Hasher};
    use core::iter::{Extend, FromIterator, IntoIterator, Iterator};

    /// The first table size: enough buckets that small maps never grow.
    const INITIAL_BUCKETS: usize = 16;

    /// The table grows when the load factor would pass this ratio.
    const MAX_LOAD_NUM: usize = 7;
    const MAX_LOAD_DEN: usize = 10;

    /// FNV-1a hasher (64-bit). Fast, simple and dependency free.
    struct FnvHasher(u64);

    impl FnvHasher {
        const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
        const PRIME: u64 = 0x100000001b3;

        fn new() -> Self {
            Self(Self::OFFSET_BASIS)
        }
    }

    impl Hasher for FnvHasher {
        fn finish(&self) -> u64 {
            self.0
        }

        fn write(&mut self, bytes: &[u8]) {
            for byte in bytes {
                self.0 ^= *byte as u64;
                self.0 = self.0.wrapping_mul(Self::PRIME);
            }
        }
    }

    /// Hash a key with FNV-1a.
    fn hash_key<K: Hash + ?Sized>(key: &K) -> u64 {
        let mut hasher = FnvHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// One bucket of the table.
    enum Slot<K, V> {
        Empty,
        Deleted,
        Occupied(K, V),
    }

    /// Where a key was found, or where it may be inserted.
    enum ProbeOutcome {
        Found(usize),
        Insert(usize),
    }

    /// A hash table mapping keys to values.
    ///
    /// Keys are hashed with FNV-1a and only need Hash and Eq; no
    /// ordering is imposed on them. Iteration order is unspecified.
    pub struct HashMap<K, V> {
        buckets: Vec<Slot<K, V>>,
        len: usize,
        deleted: usize,
    }

    impl<K, V> HashMap<K, V> {
        /// Creates an empty map.
        pub fn new() -> Self {
            Self::with_capacity(0)
        }

        /// Creates an empty map able to hold at least capacity entries
        /// without growing.
        pub fn with_capacity(capacity: usize) -> Self {
            let mut buckets = INITIAL_BUCKETS;
            while buckets * MAX_LOAD_NUM / MAX_LOAD_DEN < capacity {
                buckets *= 2;
            }
            let mut vec = Vec::with_capacity(buckets);
            for _ in 0..buckets {
                vec.push(Slot::Empty);
            }
            Self {
                buckets: vec,
                len: 0,
                deleted: 0,
            }
        }

        /// The number of entries in the map.
        pub fn len(&self) -> usize {
            self.len
        }

        /// Whether the map holds no entries.
        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// How many entries the map can hold without growing.
        pub fn capacity(&self) -> usize {
            self.buckets.len() * MAX_LOAD_NUM / MAX_LOAD_DEN
        }

        /// Removes all entries, keeping the current bucket count.
        pub fn clear(&mut self) {
            for slot in self.buckets.iter_mut() {
                *slot = Slot::Empty;
            }
            self.len = 0;
            self.deleted = 0;
        }

        /// An iterator over the entries, in no particular order.
        pub fn iter(&self) -> Iter<'_, K, V> {
            Iter {
                inner: self.buckets.iter(),
                remaining: self.len,
            }
        }

        /// An iterator over the entries, with mutable values.
        pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
            IterMut {
                inner: self.buckets.iter_mut(),
                remaining: self.len,
            }
        }

        /// An iterator over the keys, in no particular order.
        pub fn keys(&self) -> Keys<'_, K, V> {
            Keys { inner: self.iter() }
        }

        /// An iterator over the values, in no particular order.
        pub fn values(&self) -> Values<'_, K, V> {
            Values { inner: self.iter() }
        }

        /// An iterator over the values, with mutable access.
        pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
            ValuesMut {
                inner: self.iter_mut(),
            }
        }
    }

    impl<K: Hash + Eq, V> HashMap<K, V> {
        /// The number of entries that fit before the table must grow.
        fn threshold(&self) -> usize {
            self.buckets.len() * MAX_LOAD_NUM / MAX_LOAD_DEN
        }

        /// The mask used to wrap probe indices around the table.
        fn mask(&self) -> usize {
            self.buckets.len() - 1
        }

        /// Finds the bucket holding key, if any.
        fn find_index<Q: ?Sized>(&self, key: &Q) -> Option<usize>
        where
            K: Borrow<Q>,
            Q: Hash + Eq,
        {
            let hash = hash_key(key);
            let mut index = (hash as usize) & self.mask();
            for _ in 0..self.buckets.len() {
                match &self.buckets[index] {
                    Slot::Empty => return None,
                    Slot::Occupied(k, _) if k.borrow() == key => return Some(index),
                    _ => {}
                }
                index = (index + 1) & self.mask();
            }
            None
        }

        /// Finds where key lives, or the first slot it may be put in
        /// (reusing a tombstone when one was seen on the way).
        fn probe_insert(&self, key: &K) -> ProbeOutcome {
            let hash = hash_key(key);
            let mut index = (hash as usize) & self.mask();
            let mut first_deleted = None;
            for _ in 0..self.buckets.len() {
                match &self.buckets[index] {
                    Slot::Empty => {
                        return ProbeOutcome::Insert(first_deleted.unwrap_or(index));
                    }
                    Slot::Deleted => {
                        if first_deleted.is_none() {
                            first_deleted = Some(index);
                        }
                    }
                    Slot::Occupied(k, _) if k == key => {
                        return ProbeOutcome::Found(index);
                    }
                    _ => {}
                }
                index = (index + 1) & self.mask();
            }
            // The load factor is capped below the table size, so a whole
            // pass without an empty bucket still leaves a tombstone.
            match first_deleted {
                Some(index) => ProbeOutcome::Insert(index),
                None => unreachable!(),
            }
        }

        /// Inserts without checking the load factor (the caller grows
        /// first, or the table is being rebuilt).
        fn insert_no_grow(&mut self, key: K, value: V) {
            match self.probe_insert(&key) {
                ProbeOutcome::Found(index) => {
                    if let Slot::Occupied(_, old) = &mut self.buckets[index] {
                        *old = value;
                    }
                }
                ProbeOutcome::Insert(index) => {
                    if matches!(self.buckets[index], Slot::Deleted) {
                        self.deleted -= 1;
                    }
                    self.buckets[index] = Slot::Occupied(key, value);
                    self.len += 1;
                }
            }
        }

        /// Puts an entry into a slot known to be empty or deleted, for
        /// the entry API, and returns a reference to the value.
        fn insert_at(&mut self, index: usize, key: K, value: V) -> &mut V {
            if matches!(self.buckets[index], Slot::Deleted) {
                self.deleted -= 1;
            }
            self.buckets[index] = Slot::Occupied(key, value);
            self.len += 1;
            if let Slot::Occupied(_, value) = &mut self.buckets[index] {
                value
            } else {
                unreachable!()
            }
        }

        /// Rebuilds the table with new_len buckets, dropping tombstones.
        fn rehash(&mut self, new_len: usize) {
            let mut new_buckets: Vec<Slot<K, V>> = Vec::with_capacity(new_len);
            for _ in 0..new_len {
                new_buckets.push(Slot::Empty);
            }
            let old_buckets = core::mem::replace(&mut self.buckets, new_buckets);
            self.len = 0;
            self.deleted = 0;
            for slot in old_buckets {
                if let Slot::Occupied(k, v) = slot {
                    self.insert_no_grow(k, v);
                }
            }
        }

        /// Grows the table (or rebuilds it to drop tombstones) when the
        /// next insert would exceed the load factor.
        fn ensure_room(&mut self) {
            if self.len + 1 > self.threshold() || self.deleted > self.buckets.len() / 4 {
                let grown = self.len + 1 > self.threshold();
                let new_len = if grown {
                    self.buckets.len() * 2
                } else {
                    self.buckets.len()
                };
                self.rehash(new_len);
            }
        }

        /// Makes room for at least additional more entries.
        pub fn reserve(&mut self, additional: usize) {
            while self.len + additional > self.threshold() {
                let new_len = self.buckets.len() * 2;
                self.rehash(new_len);
            }
            if self.deleted > self.buckets.len() / 4 {
                let same_len = self.buckets.len();
                self.rehash(same_len);
            }
        }

        /// Inserts a key-value pair, returning the old value if the key
        /// was already present.
        pub fn insert(&mut self, key: K, value: V) -> Option<V> {
            self.ensure_room();
            match self.probe_insert(&key) {
                ProbeOutcome::Found(index) => {
                    if let Slot::Occupied(_, old) = &mut self.buckets[index] {
                        Some(core::mem::replace(old, value))
                    } else {
                        unreachable!()
                    }
                }
                ProbeOutcome::Insert(index) => {
                    if matches!(self.buckets[index], Slot::Deleted) {
                        self.deleted -= 1;
                    }
                    self.buckets[index] = Slot::Occupied(key, value);
                    self.len += 1;
                    None
                }
            }
        }

        /// Returns a reference to the value for key, if it is present.
        pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V>
        where
            K: Borrow<Q>,
            Q: Hash + Eq,
        {
            self.find_index(key).map(|index| {
                if let Slot::Occupied(_, v) = &self.buckets[index] {
                    v
                } else {
                    unreachable!()
                }
            })
        }

        /// Returns a mutable reference to the value for key.
        pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
        where
            K: Borrow<Q>,
            Q: Hash + Eq,
        {
            self.find_index(key).map(|index| {
                if let Slot::Occupied(_, v) = &mut self.buckets[index] {
                    v
                } else {
                    unreachable!()
                }
            })
        }

        /// Whether key is in the map.
        pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
        where
            K: Borrow<Q>,
            Q: Hash + Eq,
        {
            self.find_index(key).is_some()
        }

        /// Removes key and returns its value, if it was present.
        pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V>
        where
            K: Borrow<Q>,
            Q: Hash + Eq,
        {
            let index = self.find_index(key)?;
            match core::mem::replace(&mut self.buckets[index], Slot::Deleted) {
                Slot::Occupied(_, value) => {
                    self.len -= 1;
                    self.deleted += 1;
                    if self.deleted > self.buckets.len() / 2 {
                        let same_len = self.buckets.len();
                        self.rehash(same_len);
                    }
                    Some(value)
                }
                _ => unreachable!(),
            }
        }

        /// Gets the entry for key, for in-place manipulation.
        pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
            self.ensure_room();
            match self.probe_insert(&key) {
                ProbeOutcome::Found(index) => {
                    Entry::Occupied(OccupiedEntry { map: self, index })
                }
                ProbeOutcome::Insert(index) => {
                    Entry::Vacant(VacantEntry { map: self, index, key })
                }
            }
        }
    }

    impl<K: Hash + Eq, V> Default for HashMap<K, V> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for HashMap<K, V> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_map().entries(self.iter()).finish()
        }
    }

    impl<K: Hash + Eq + Clone, V: Clone> Clone for HashMap<K, V> {
        fn clone(&self) -> Self {
            let mut map = Self::with_capacity(self.capacity());
            for (k, v) in self.iter() {
                map.insert(k.clone(), v.clone());
            }
            map
        }
    }

    impl<K: Hash + Eq, V> Extend<(K, V)> for HashMap<K, V> {
        fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
            for (k, v) in iter {
                self.insert(k, v);
            }
        }
    }

    impl<K: Hash + Eq, V> FromIterator<(K, V)> for HashMap<K, V> {
        fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
            let iter = iter.into_iter();
            let mut map = Self::with_capacity(iter.size_hint().0);
            for (k, v) in iter {
                map.insert(k, v);
            }
            map
        }
    }

    /// An iterator over the entries of a HashMap.
    pub struct Iter<'a, K, V> {
        inner: core::slice::Iter<'a, Slot<K, V>>,
        remaining: usize,
    }

    impl<'a, K, V> Iterator for Iter<'a, K, V> {
        type Item = (&'a K, &'a V);

        fn next(&mut self) -> Option<Self::Item> {
            while let Some(slot) = self.inner.next() {
                if let Slot::Occupied(k, v) = slot {
                    self.remaining -= 1;
                    return Some((k, v));
                }
            }
            None
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.remaining, Some(self.remaining))
        }
    }

    /// An iterator over the entries of a HashMap, with mutable values.
    pub struct IterMut<'a, K, V> {
        inner: core::slice::IterMut<'a, Slot<K, V>>,
        remaining: usize,
    }

    impl<'a, K, V> Iterator for IterMut<'a, K, V> {
        type Item = (&'a K, &'a mut V);

        fn next(&mut self) -> Option<Self::Item> {
            while let Some(slot) = self.inner.next() {
                if let Slot::Occupied(k, v) = slot {
                    self.remaining -= 1;
                    return Some((&*k, v));
                }
            }
            None
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.remaining, Some(self.remaining))
        }
    }

    /// An iterator over the keys of a HashMap.
    pub struct Keys<'a, K, V> {
        inner: Iter<'a, K, V>,
    }

    impl<'a, K, V> Iterator for Keys<'a, K, V> {
        type Item = &'a K;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next().map(|(k, _)| k)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }

    /// An iterator over the values of a HashMap.
    pub struct Values<'a, K, V> {
        inner: Iter<'a, K, V>,
    }

    impl<'a, K, V> Iterator for Values<'a, K, V> {
        type Item = &'a V;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next().map(|(_, v)| v)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }

    /// An iterator over the values of a HashMap, with mutable access.
    pub struct ValuesMut<'a, K, V> {
        inner: IterMut<'a, K, V>,
    }

    impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
        type Item = &'a mut V;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next().map(|(_, v)| v)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.inner.size_hint()
        }
    }

    /// An owning iterator over the entries of a HashMap.
    pub struct IntoIter<K, V> {
        inner: VecIntoIter<Slot<K, V>>,
        remaining: usize,
    }

    impl<K, V> Iterator for IntoIter<K, V> {
        type Item = (K, V);

        fn next(&mut self) -> Option<Self::Item> {
            while let Some(slot) = self.inner.next() {
                if let Slot::Occupied(k, v) = slot {
                    self.remaining -= 1;
                    return Some((k, v));
                }
            }
            None
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.remaining, Some(self.remaining))
        }
    }

    impl<K, V> IntoIterator for HashMap<K, V> {
        type Item = (K, V);
        type IntoIter = IntoIter<K, V>;

        fn into_iter(self) -> IntoIter<K, V> {
            let len = self.len;
            IntoIter {
                inner: self.buckets.into_iter(),
                remaining: len,
            }
        }
    }

    impl<'a, K, V> IntoIterator for &'a HashMap<K, V> {
        type Item = (&'a K, &'a V);
        type IntoIter = Iter<'a, K, V>;

        fn into_iter(self) -> Iter<'a, K, V> {
            self.iter()
        }
    }

    impl<'a, K, V> IntoIterator for &'a mut HashMap<K, V> {
        type Item = (&'a K, &'a mut V);
        type IntoIter = IterMut<'a, K, V>;

        fn into_iter(self) -> IterMut<'a, K, V> {
            self.iter_mut()
        }
    }

    /// A view into a single entry in a map, for in-place manipulation.
    pub enum Entry<'a, K, V> {
        Occupied(OccupiedEntry<'a, K, V>),
        Vacant(VacantEntry<'a, K, V>),
    }

    impl<'a, K: Hash + Eq, V> Entry<'a, K, V> {
        /// The key of this entry.
        pub fn key(&self) -> &K {
            match self {
                Entry::Occupied(entry) => entry.key(),
                Entry::Vacant(entry) => entry.key(),
            }
        }

        /// Modifies the value if the entry is occupied.
        pub fn and_modify<F: FnOnce(&mut V)>(self, f: F) -> Self {
            match self {
                Entry::Occupied(mut entry) => {
                    f(entry.get_mut());
                    Entry::Occupied(entry)
                }
                Entry::Vacant(entry) => Entry::Vacant(entry),
            }
        }

        /// Returns the value, inserting default if the entry is vacant.
        pub fn or_insert(self, default: V) -> &'a mut V {
            match self {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(default),
            }
        }

        /// Returns the value, inserting the result of f if vacant.
        pub fn or_insert_with<F: FnOnce() -> V>(self, f: F) -> &'a mut V {
            match self {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let value = f();
                    entry.insert(value)
                }
            }
        }

        /// Returns the value, inserting the result of f(key) if vacant.
        pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, f: F) -> &'a mut V {
            match self {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert_with(f),
            }
        }
    }

    /// A view into an occupied entry in a map.
    pub struct OccupiedEntry<'a, K, V> {
        map: &'a mut HashMap<K, V>,
        index: usize,
    }

    impl<'a, K: Hash + Eq, V> OccupiedEntry<'a, K, V> {
        /// The key of this entry.
        pub fn key(&self) -> &K {
            if let Slot::Occupied(k, _) = &self.map.buckets[self.index] {
                k
            } else {
                unreachable!()
            }
        }

        /// A reference to the value of this entry.
        pub fn get(&self) -> &V {
            if let Slot::Occupied(_, v) = &self.map.buckets[self.index] {
                v
            } else {
                unreachable!()
            }
        }

        /// A mutable reference to the value of this entry.
        pub fn get_mut(&mut self) -> &mut V {
            if let Slot::Occupied(_, v) = &mut self.map.buckets[self.index] {
                v
            } else {
                unreachable!()
            }
        }

        /// Converts the entry into a mutable reference to its value.
        pub fn into_mut(self) -> &'a mut V {
            if let Slot::Occupied(_, v) = &mut self.map.buckets[self.index] {
                v
            } else {
                unreachable!()
            }
        }

        /// Replaces the value, returning the old one.
        pub fn insert(&mut self, value: V) -> V {
            if let Slot::Occupied(_, old) = &mut self.map.buckets[self.index] {
                core::mem::replace(old, value)
            } else {
                unreachable!()
            }
        }

        /// Removes the entry, returning its value.
        pub fn remove(self) -> V {
            let slot = core::mem::replace(&mut self.map.buckets[self.index], Slot::Deleted);
            if let Slot::Occupied(_, value) = slot {
                self.map.len -= 1;
                self.map.deleted += 1;
                value
            } else {
                unreachable!()
            }
        }
    }

    /// A view into a vacant entry in a map.
    pub struct VacantEntry<'a, K, V> {
        map: &'a mut HashMap<K, V>,
        key: K,
        index: usize,
    }

    impl<'a, K: Hash + Eq, V> VacantEntry<'a, K, V> {
        /// The key that would be used when inserting.
        pub fn key(&self) -> &K {
            &self.key
        }

        /// The key that would be used when inserting.
        pub fn into_key(self) -> K {
            self.key
        }

        /// Inserts a value and returns a reference to it.
        pub fn insert(self, value: V) -> &'a mut V {
            let VacantEntry { map, key, index } = self;
            map.insert_at(index, key, value)
        }

        /// Inserts the result of f(key) and returns a reference to it.
        fn insert_with<F: FnOnce(&K) -> V>(self, f: F) -> &'a mut V {
            let value = f(&self.key);
            self.insert(value)
        }

        /// Inserts default and returns a reference to it.
        pub fn or_insert(self, default: V) -> &'a mut V {
            self.insert(default)
        }

        /// Inserts the result of f() and returns a reference to it.
        pub fn or_insert_with<F: FnOnce() -> V>(self, f: F) -> &'a mut V {
            let value = f();
            self.insert(value)
        }

        /// Inserts the result of f(key) and returns a reference to it.
        pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, f: F) -> &'a mut V {
            self.insert_with(f)
        }
    }
}

#[doc(no_inline)]
pub use core::{arch, cell, cmp, hint, marker, mem, ops, ptr, slice, str};

#[macro_use]
mod macros;

pub mod env;
pub mod io;
pub mod os;
pub mod process;
pub mod sync;
pub mod thread;
pub mod time;

#[cfg(feature = "fs")]
pub mod fs;
#[cfg(feature = "net")]
pub mod net;
