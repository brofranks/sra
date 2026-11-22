//! Logical sorts and their constructor and projection declarations.
//!
//! The declaration store pins canonical sorts. Structural dependencies are
//! traced edges, and constructor and projection identities follow from the
//! declaring sort and the member position. Logical sorts describe values
//! independently of their physical representation.

use std::collections::HashMap;
use std::fmt;
use std::sync::{LazyLock, Mutex};

use crate::idx;
use crate::scalar::ScalarSort;
use crate::util;
use crate::util::dag::{Dag, Exhausted, Grow, Shared, structural_hash};

idx! {
    // One pinned sort node in its declaration store.
    struct SortId;
}

/// The shape of a logical sort.
///
/// Aggregate types stay above the scalar backend boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SortKind {
    /// A scalar with its complete backend universe.
    Scalar(ScalarSort),
    /// A structural product of member sorts.
    Product,
    /// A nominal record with named members.
    Record,
}

/// A logical sort in one declaration store.
///
/// A copy describes the same sort and does not retain its store. Cached
/// metadata gives the sort's shape and scalar-leaf count directly.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Sort {
    owner: u64,
    id: SortId,
    kind: SortKind,
    len: usize,
}

impl Sort {
    /// The shape of this sort.
    pub const fn kind(self) -> SortKind {
        self.kind
    }

    /// The flattened scalar-leaf count, one for a scalar.
    pub const fn scalar_len(self) -> usize {
        self.len
    }

    /// The scalar universe used by backends.
    ///
    /// Symbol namespaces share the universe returned by
    /// [`ScalarSort::symbol`]. The logical [`Sort`] identifies the namespace.
    ///
    /// # Panics
    ///
    /// Panics if this sort is an aggregate.
    pub const fn scalar(self) -> ScalarSort {
        let SortKind::Scalar(sort) = self.kind else {
            panic!("expected a scalar sort");
        };
        sort
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Named {
    name: Box<str>,
    labels: Box<[Box<str>]>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Node {
    kind: SortKind,
    len: usize,
    named: Option<Box<Named>>,
}

impl Node {
    fn sort(&self, owner: u64, id: SortId) -> Sort {
        Sort {
            owner,
            id,
            kind: self.kind,
            len: self.len,
        }
    }

    fn members<'a>(&self, edges: &'a [SortId]) -> &'a [SortId] {
        // A record's final edge holds the canonical product of its arguments.
        if self.kind == SortKind::Record {
            &edges[..edges.len() - 1]
        } else {
            edges
        }
    }
}

type State = Dag<SortId, Node, SortId>;

impl Grow for State {
    type Cx<'a> = ();

    fn grow(&mut self, exhausted: Exhausted, _: ()) {
        // Declarations are few and every canonical node is pinned, so
        // maintenance runs on the calling thread.
        self.maintain(
            exhausted,
            1,
            |&child| Some(child),
            |_| {},
            |node, edges| structural_hash((node, edges)),
        );
    }
}

/// The logical signature of a constructor or projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FuncInfo {
    /// The input sort of the declaration.
    ///
    /// A constructor uses [`Sorts::product`] over the record's immediate
    /// member sorts. A projection uses its declaring sort.
    pub domain: Sort,
    /// The result sort.
    pub range: Sort,
    /// The projected member position, or `None` for a constructor.
    pub member: Option<usize>,
}

/// A constructor or projection with a fixed logical signature.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FuncDecl {
    owner: u64,
    parent: SortId,
    member: Option<usize>,
}

impl FuncDecl {
    /// The signature in the owning declaration store.
    ///
    /// # Panics
    ///
    /// Panics if this declaration belongs to another store.
    pub fn resolve(self, sorts: &Sorts) -> FuncInfo {
        sorts.check_owner(self.owner);
        let state = sorts.state.read(());
        let node = state.node(self.parent);
        let parent = node.sort(self.owner, self.parent);
        let (domain, range) = if let Some(i) = self.member {
            let id = node.edges[i];
            (parent, state.node(id).sort(self.owner, id))
        } else {
            let id = *node
                .edges
                .last()
                .expect("a record must store its argument product");
            (state.node(id).sort(self.owner, id), parent)
        };
        FuncInfo {
            domain,
            range,
            member: self.member,
        }
    }

    /// The record name, record member name, or `project{i}` for a product.
    ///
    /// Projection names can repeat. The declaring sort and member position
    /// identify each projection.
    ///
    /// # Panics
    ///
    /// Panics if this declaration belongs to another store.
    pub fn name(self, sorts: &Sorts) -> String {
        sorts.check_owner(self.owner);
        let state = sorts.state.read(());
        let node = state.node(self.parent);
        match (&node.named, self.member) {
            (Some(named), None) => named.name.to_string(),
            (Some(named), Some(i)) => named.labels[i].to_string(),
            (None, Some(i)) => format!("project{i}"),
            (None, None) => unreachable!("constructor requires a named record"),
        }
    }
}

/// The store that owns logical sort definitions and gives them identity.
///
/// Canonical sorts remain valid for the lifetime of the store. Calls can
/// declare sorts through a shared reference. Records and symbol namespaces
/// share one name table.
pub struct Sorts {
    id: u64,
    state: LazyLock<Shared<State>>,
    named: Mutex<HashMap<Box<str>, SortId>>,
}

impl Sorts {
    /// The unique owner identity shared by this store's declarations.
    pub const fn id(&self) -> u64 {
        self.id
    }

    fn check_owner(&self, owner: u64) {
        assert!(owner == self.id, "declaration belongs to another store");
    }

    fn intern(&self, node: Node, edges: &[SortId]) -> Sort {
        let hash = structural_hash((&node, edges));
        let id = self.state.retry((), |state| {
            let id = state.intern(
                hash,
                |value, children| (value, children) == (&node, edges),
                || node.clone(),
                edges,
            )?;
            // The canonical result lives as long as the store. Losing
            // candidates remain eligible for collection.
            state.pin(id);
            Ok(id)
        });
        node.sort(self.id, id)
    }

    fn declare(&self, kind: SortKind, name: &str, members: &[(&str, Sort)], len: usize) -> Sort {
        let mut named = self.named.lock().expect("name table must not be poisoned");
        if let Some(&id) = named.get(name) {
            // Identical definitions share this node. Release the lock so a
            // conflicting definition leaves the name table usable.
            drop(named);
            let state = self.state.read(());
            let node = state.node(id);
            let labels = &node
                .named
                .as_ref()
                .expect("name table must map to named sorts")
                .labels;
            let same = node.kind == kind
                && labels
                    .iter()
                    .map(Box::as_ref)
                    .zip(node.members(node.edges).iter().copied())
                    .eq(members.iter().map(|&(label, sort)| (label, sort.id)));
            assert!(same, "conflicting definition for named sort {name}");
            return node.sort(self.id, id);
        }

        let args = (kind == SortKind::Record)
            .then(|| self.product(members.iter().map(|&(_, sort)| sort)).id);
        let edges: Vec<_> = members
            .iter()
            .map(|&(_, sort)| sort.id)
            .chain(args)
            .collect();
        let sort = self.intern(
            Node {
                kind,
                len,
                named: Some(Box::new(Named {
                    name: name.into(),
                    labels: members.iter().map(|&(label, _)| label.into()).collect(),
                })),
            },
            &edges,
        );
        named.insert(name.into(), sort.id);
        sort
    }

    fn scalar(&self, sort: ScalarSort) -> Sort {
        self.intern(
            Node {
                kind: SortKind::Scalar(sort),
                len: 1,
                named: None,
            },
            &[],
        )
    }

    /// The interned Boolean sort, distinct from a one-bit bit-vector.
    pub fn bool(&self) -> Sort {
        self.scalar(ScalarSort::bool())
    }

    /// The interned bit-vector sort for `width`.
    ///
    /// # Panics
    ///
    /// Panics if `width` is zero.
    pub fn bv(&self, width: u32) -> Sort {
        self.scalar(ScalarSort::bv(width))
    }

    /// Declares a named symbol namespace over all 64-bit IDs.
    ///
    /// # Panics
    ///
    /// Panics if `name` names a conflicting definition.
    pub fn symbol(&self, name: &str) -> Sort {
        // A symbol namespace has no arguments and is one scalar leaf.
        self.declare(SortKind::Scalar(ScalarSort::symbol()), name, &[], 1)
    }

    /// The interned anonymous product of `sorts`.
    ///
    /// Anonymous products are flat, with unit removed and singletons
    /// collapsed. Named records remain whole components even when they contain
    /// no scalar leaves.
    ///
    /// # Panics
    ///
    /// Panics if a sort belongs to another store or the scalar-leaf count
    /// overflows.
    pub fn product(&self, sorts: impl IntoIterator<Item = Sort>) -> Sort {
        let mut edges = Vec::new();
        let mut len = Some(0_usize);
        for sort in sorts {
            self.check_owner(sort.owner);
            len = len.and_then(|n| n.checked_add(sort.len));
            if sort.kind == SortKind::Product {
                let state = self.state.read(());
                edges.extend_from_slice(state.node(sort.id).edges);
            } else {
                edges.push(sort.id);
            }
        }
        if let &[id] = edges.as_slice() {
            return self.state.read(()).node(id).sort(self.id, id);
        }
        self.intern(
            Node {
                kind: SortKind::Product,
                len: len.expect("scalar leaf count overflow"),
                named: None,
            },
            &edges,
        )
    }

    /// The empty product.
    pub fn unit(&self) -> Sort {
        self.product([])
    }

    /// Declares or retrieves an identical nominal record.
    ///
    /// The definition includes member names and sorts in declaration order.
    ///
    /// # Panics
    ///
    /// Panics if a member sort belongs to another store, the scalar-leaf count
    /// overflows, or `name` names a conflicting definition.
    pub fn record(&self, name: &str, members: &[(&str, Sort)]) -> Sort {
        let len = members
            .iter()
            .try_fold(0_usize, |len, &(_, sort)| {
                self.check_owner(sort.owner);
                len.checked_add(sort.len)
            })
            .expect("scalar leaf count overflow");
        self.declare(SortKind::Record, name, members, len)
    }

    /// The immediate member sorts in declaration order.
    ///
    /// Scalars have no members.
    ///
    /// # Panics
    ///
    /// Panics if `sort` belongs to another store.
    pub fn members(&self, sort: Sort) -> Vec<Sort> {
        self.check_owner(sort.owner);
        let state = self.state.read(());
        let node = state.node(sort.id);
        node.members(node.edges)
            .iter()
            .map(|&id| state.node(id).sort(self.id, id))
            .collect()
    }

    /// The constructor from the argument product to a record sort.
    ///
    /// # Panics
    ///
    /// Panics if `sort` belongs to another store or is not a record.
    pub fn constructor(&self, sort: Sort) -> FuncDecl {
        self.check_owner(sort.owner);
        assert!(
            sort.kind == SortKind::Record,
            "constructor requires a record"
        );
        FuncDecl {
            owner: self.id,
            parent: sort.id,
            member: None,
        }
    }

    /// The projection of member `i` from a product or record.
    ///
    /// # Panics
    ///
    /// Panics if `sort` belongs to another store, is a scalar, or has no
    /// member at `i`.
    pub fn project(&self, sort: Sort, i: usize) -> FuncDecl {
        self.check_owner(sort.owner);
        assert!(
            !matches!(sort.kind, SortKind::Scalar(_)),
            "projection requires an aggregate sort"
        );
        let state = self.state.read(());
        let node = state.node(sort.id);
        assert!(
            i < node.members(node.edges).len(),
            "projection index must name a component of the sort"
        );
        FuncDecl {
            owner: self.id,
            parent: sort.id,
            member: Some(i),
        }
    }
}

impl Default for Sorts {
    /// An empty declaration store with a fresh identity.
    fn default() -> Self {
        Self {
            id: util::fresh_id(),
            state: LazyLock::new(|| Shared::new(State::with_capacity(256, 512, []))),
            named: Mutex::default(),
        }
    }
}

impl fmt::Debug for Sorts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sorts")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "../tests/unit/ast.rs"]
mod tests;
