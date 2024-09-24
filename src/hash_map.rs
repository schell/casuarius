pub use self::imp::*;

#[cfg(feature = "std")]
mod imp {
    pub use rustc_hash::{FxHashMap, FxHashSet};
    pub use std::collections::hash_map::Entry;
}

#[cfg(not(feature = "std"))]
mod imp {
    use rustc_hash::FxHasher;

    pub use hashbrown::hash_map::Entry;

    pub type FxHashMap<K, V> = hashbrown::HashMap<K, V, FxBuildHasher>;
    pub type FxHashSet<V> = hashbrown::HashSet<V, FxBuildHasher>;

    #[derive(Copy, Clone, Default)]
    pub struct FxBuildHasher;

    impl core::hash::BuildHasher for FxBuildHasher {
        type Hasher = FxHasher;
        fn build_hasher(&self) -> FxHasher {
            rustc_hash::FxHasher::default()
        }
    }
}
