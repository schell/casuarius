//! This crate contains an implementation of the Cassowary constraint solving algorithm, based upon the work by
//! G.J. Badros et al. in 2001. This algorithm is designed primarily for use constraining elements in
//! user interfaces, but works well for many constraints that use floats.
//! Constraints are linear combinations of the problem variables. The notable features of Cassowary that make it
//! ideal for user interfaces are that it is incremental (i.e. you can add and remove constraints at runtime
//! and it will perform the minimum work to update the result) and that the constraints can be violated if
//! necessary,
//! with the order in which they are violated specified by setting a "strength" for each constraint.
//! This allows the solution to gracefully degrade, which is useful for when a
//! user interface needs to compromise on its constraints in order to still be able to display something.
//!
//! ## Constraint builder
//!
//! This crate aims to provide a builder for describing linear constraints as naturally as possible.
//!
//! For example, for the constraint
//! `(a + b) * 2 + c >= d + 1` with strength `s`, the code to use is
//!
//! ```ignore
//! ((a + b) * 2.0 + c)
//!     .is_ge(d + 1.0)
//!     .with_strength(s)
//! ```
//!
//! This crate also provides the `derive_syntax_for` macro, which allows you to use your own named
//! variables.
//!
//! # A simple example
//!
//! Imagine a layout consisting of two elements laid out horizontally. For small window widths the elements
//! should compress to fit, but if there is enough space they should display at their preferred widths. The
//! first element will align to the left, and the second to the right. For  this example we will ignore
//! vertical layout.
//!
//! ```rust
//! use casuarius::*;
//!
//! // We define the variables required using an Element type with left
//! // and right edges, and the width of the window.
//!
//! struct Element {
//!     left: Variable,
//!     right: Variable
//! }
//! let box1 = Element {
//!     left: Variable("box1.left"),
//!     right: Variable("box1.right")
//! };
//!
//! let window_width = Variable("window_width");
//!
//! let box2 = Element {
//!     left: Variable("box2.left"),
//!     right: Variable("box2.right")
//! };
//!
//! // Now we set up the solver and constraints.
//!
//! let mut solver = Solver::<Variable>::default();
//! solver.add_constraints(vec![
//!     window_width.is_ge(0.0), // positive window width
//!     box1.left.is(0.0), // left align
//!     box2.right.is(window_width), // right align
//!     box2.left.is_ge(box1.right), // no overlap
//!     // positive widths
//!     box1.left.is_le(box1.right),
//!     box2.left.is_le(box2.right),
//!     // preferred widths:
//!     (box1.right - box1.left).is(50.0).with_strength(WEAK),
//!     (box2.right - box2.left).is(100.0).with_strength(WEAK)
//! ]).unwrap();
//!
//! // The window width is currently free to take any positive value. Let's constrain it to a particular value.
//! // Since for this example we will repeatedly change the window width, it is most efficient to use an
//! // "edit variable", instead of repeatedly removing and adding constraints (note that for efficiency
//! // reasons we cannot edit a normal constraint that has been added to the solver).
//!
//! solver.add_edit_variable(window_width, STRONG).unwrap();
//! solver.suggest_value(window_width, 300.0).unwrap();
//!
//! // This value of 300 is enough to fit both boxes in with room to spare, so let's check that this is the case.
//! // We can fetch a list of changes to the values of variables in the solver. Using the pretty printer defined
//! // earlier we can see what values our variables now hold.
//!
//! let mut print_changes = || {
//!     println!("Changes:");
//!     solver
//!         .fetch_changes()
//!         .iter()
//!         .map(|(var, val)| println!("{}: {}", var.0, val));
//! };
//! print_changes();
//!
//! // Changes:
//! // window_width: 300
//! // box1.left -0
//! // box1.right 50
//! // box2.left 200
//! // box2.right 300
//!
//! // Note that the value of `box1.left` is not mentioned. This is because `solver.fetch_changes` only lists
//! // *changes* to variables, and since each variable starts in the solver with a value of zero, any values that
//! // have not changed from zero will not be reported.
//!
//! // Just to be thorough, let's assert our current values:
//! let ww = solver.get_value(window_width);
//! let b1l = solver.get_value(box1.left);
//! let b1r = solver.get_value(box1.right);
//! let b2l = solver.get_value(box2.left);
//! let b2r = solver.get_value(box2.right);
//! println!("window_width: {}", ww);
//! println!("box1.left {}", b1l);
//! println!("box1.right {}", b1r);
//! println!("box2.left {}", b2l);
//! println!("box2.right {}", b2r);
//! assert!(ww >= 0.0);
//! assert_eq!(0.0, b1l);
//! assert_eq!(ww, b2r, "box2.right ({}) != ww ({})", b2r, ww);
//! assert!(b2l >= b1r);
//! assert!(b1l <= b1r);
//! assert!(b2l <= b2r);
//! assert_eq!(50.0, b1r - b1l, "box1 width");
//! assert_eq!(100.0, b2r - b2l, "box2 width");
//!
//! // Now let's try compressing the window so that the boxes can't take up their preferred widths.
//!
//! solver.suggest_value(window_width, 75.0);
//!
//! // Now the solver can't satisfy all of the constraints. It will pick at least one of the weakest
//! // constraints to violate. In this case it will be one or both of the preferred widths.
//!
//! let expected_changes = vec![
//!     (box2.right, 75.0),
//!     (box2.left, 0.0),
//!     (box1.right, 0.0),
//!     (window_width, 75.0),
//! ];
//! let changes = solver.fetch_changes().iter().copied().collect::<Vec<_>>();
//!
//! assert_eq!(expected_changes, changes);
//!
//! // In a user interface this is not likely a result we would prefer. The solution is to add another constraint
//! // to control the behaviour when the preferred widths cannot both be satisfied. In this example we are going
//! // to constrain the boxes to try to maintain a ratio between their widths.
//!
//! solver.add_constraint(
//!     ((box1.right - box1.left) / 50.0f64)
//!         .is((box2.right - box2.left) / 100.0f64)
//! ).unwrap();
//!
//! // Now the result gives values that maintain the ratio between the sizes of the two boxes:
//!
//! let box1_width = solver.get_value(box1.right) - solver.get_value(box1.left);
//! let box2_width = solver.get_value(box2.right) - solver.get_value(box2.left);
//! assert_eq!(box1_width / 50.0, box2_width / 100.0, "box width ratios");
//! ```
//!
//! This example may have appeared somewhat contrived, but hopefully it shows the power of the cassowary
//! algorithm for laying out user interfaces.
//!
//! One thing that this example exposes is that this crate is a rather low level library. It does not have
//! any inherent knowledge of user interfaces, directions or boxes. Thus for use in a user interface this
//! crate should ideally be wrapped by a higher level API, which is outside the scope of this crate.
use std::collections::hash_map::Entry;

use crate as casuarius;
use ordered_float::OrderedFloat;

mod operators;
mod solver_impl;
pub use operators::Constrainable;
use rustc_hash::FxHashMap;
pub use strength::{MEDIUM, REQUIRED, STRONG, WEAK};

#[macro_use]
pub mod derive_syntax;

#[cfg(doctest)]
pub mod doctest {
    #[doc = include_str!("../README.md")]
    pub struct ReadmeDoctests;
}

/// A generic variable that can be created with a `&'static str`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variable(pub &'static str);
derive_syntax_for!(Variable);

/// A variable and a coefficient to multiply that variable by. This is a sub-expression in
/// a constraint equation.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Term<T> {
    pub variable: T,
    pub coefficient: OrderedFloat<f64>,
}

impl<T> Term<T> {
    /// Construct a new Term from a variable and a coefficient.
    pub fn new(variable: T, coefficient: f64) -> Term<T> {
        Term {
            variable,
            coefficient: coefficient.into(),
        }
    }
}

/// An expression that can be the left hand or right hand side of a constraint equation.
/// It is a linear combination of variables, i.e. a sum of variables weighted by coefficients, plus an optional constant.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Expression<T> {
    pub terms: Vec<Term<T>>,
    pub constant: OrderedFloat<f64>,
}

impl<T: Clone> Expression<T> {
    /// Constructs an expression of the form _n_, where n is a constant real number, not a variable.
    pub fn from_constant(v: f64) -> Expression<T> {
        Expression {
            terms: Vec::new(),
            constant: v.into(),
        }
    }
    /// Constructs an expression from a single term. Forms an expression of the form _n x_
    /// where n is the coefficient, and x is the variable.
    pub fn from_term(term: Term<T>) -> Expression<T> {
        Expression {
            terms: vec![term],
            constant: 0.0.into(),
        }
    }
    /// General constructor. Each `Term` in `terms` is part of the sum forming the expression, as well as `constant`.
    pub fn new(terms: Vec<Term<T>>, constant: f64) -> Expression<T> {
        Expression {
            terms,
            constant: constant.into(),
        }
    }
    /// Mutates this expression by multiplying it by minus one.
    pub fn negate(&mut self) {
        self.constant = (-(self.constant.into_inner())).into();
        for t in &mut self.terms {
            let t2 = t.clone();
            *t = -t2;
        }
    }
}

impl<Var: Clone> Constrainable<Var> for Expression<Var> {
    fn equal_to<X>(self, x: X) -> Constraint<Var>
    where
        X: Into<Expression<Var>> + Clone,
    {
        let lhs = PartialConstraint(self, WeightedRelation::EQ(strength::REQUIRED));
        let rhs: Expression<Var> = x.into();
        let (op, s) = lhs.1.into();
        Constraint::new(lhs.0 - rhs, op, s)
    }

    fn greater_than_or_equal_to<X>(self, x: X) -> Constraint<Var>
    where
        X: Into<Expression<Var>> + Clone,
    {
        let lhs = PartialConstraint(self, WeightedRelation::GE(strength::REQUIRED));
        let rhs: Expression<Var> = x.into();
        let (op, s) = lhs.1.into();
        Constraint::new(lhs.0 - rhs, op, s)
    }

    fn less_than_or_equal_to<X>(self, x: X) -> Constraint<Var>
    where
        X: Into<Expression<Var>> + Clone,
    {
        let lhs = PartialConstraint(self, WeightedRelation::LE(strength::REQUIRED));
        let rhs: Expression<Var> = x.into();
        let (op, s) = lhs.1.into();
        Constraint::new(lhs.0 - rhs, op, s)
    }
}

impl<T: Clone> From<f64> for Expression<T> {
    fn from(v: f64) -> Expression<T> {
        Expression::from_constant(v)
    }
}

impl<T: Clone> From<i32> for Expression<T> {
    fn from(v: i32) -> Expression<T> {
        Expression::from_constant(v as f64)
    }
}

impl<T: Clone> From<u32> for Expression<T> {
    fn from(v: u32) -> Expression<T> {
        Expression::from_constant(v as f64)
    }
}

impl<T: Clone> From<Term<T>> for Expression<T> {
    fn from(t: Term<T>) -> Expression<T> {
        Expression::from_term(t)
    }
}

/// Contains useful constants and functions for producing strengths for use in the constraint solver.
/// Each constraint added to the solver has an associated strength specifying the precedence the solver should
/// impose when choosing which constraints to enforce. It will try to enforce all constraints, but if that
/// is impossible the lowest strength constraints are the first to be violated.
///
/// Strengths are simply real numbers. The strongest legal strength is 1,001,001,000.0. The weakest is 0.0.
/// For convenience constants are declared for commonly used strengths. These are `REQUIRED`, `STRONG`,
/// `MEDIUM` and `WEAK`. Feel free to multiply these by other values to get intermediate strengths.
/// Note that the solver will clip given strengths to the legal range.
///
/// `REQUIRED` signifies a constraint that cannot be violated under any circumstance. Use this special strength
/// sparingly, as the solver will fail completely if it find that not all of the `REQUIRED` constraints
/// can be satisfied. The other strengths represent fallible constraints. These should be the most
/// commonly used strenghts for use cases where violating a constraint is acceptable or even desired.
///
/// The solver will try to get as close to satisfying the constraints it violates as possible, strongest first.
/// This behaviour can be used (for example) to provide a "default" value for a variable should no other
/// stronger constraints be put upon it.
pub mod strength {
    /// Create a constraint as a linear combination of STRONG, MEDIUM and WEAK strengths, corresponding to `a`
    /// `b` and `c` respectively. The result is further multiplied by `w`.
    pub fn create(a: f64, b: f64, c: f64, w: f64) -> f64 {
        (a * w).clamp(0.0, 1000.0) * 1_000_000.0
            + (b * w).clamp(0.0, 1000.0) * 1000.0
            + (c * w).clamp(0.0, 1000.0)
    }
    pub const REQUIRED: f64 = 1_001_001_000.0;
    pub const STRONG: f64 = 1_000_000.0;
    pub const MEDIUM: f64 = 1_000.0;
    pub const WEAK: f64 = 1.0;

    /// Clips a strength value to the legal range
    pub fn clip(s: f64) -> f64 {
        s.clamp(0.0, REQUIRED)
    }
}

/// The possible relations that a constraint can specify.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum RelationalOperator {
    /// `<=`
    LessOrEqual,
    /// `==`
    Equal,
    /// `>=`
    GreaterOrEqual,
}

impl std::fmt::Display for RelationalOperator {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            RelationalOperator::LessOrEqual => write!(fmt, "<=")?,
            RelationalOperator::Equal => write!(fmt, "==")?,
            RelationalOperator::GreaterOrEqual => write!(fmt, ">=")?,
        };
        Ok(())
    }
}

/// A constraint, consisting of an equation governed by an expression and a relational operator,
/// and an associated strength.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Constraint<T> {
    expression: Expression<T>,
    strength: OrderedFloat<f64>,
    op: RelationalOperator,
}

impl<T> Constraint<T> {
    /// Construct a new constraint from an expression, a relational operator and a strength.
    /// This corresponds to the equation `e op 0.0`, e.g. `x + y >= 0.0`. For equations with a non-zero
    /// right hand side, subtract it from the equation to give a zero right hand side.
    pub fn new(e: Expression<T>, op: RelationalOperator, strength: f64) -> Constraint<T> {
        Constraint {
            expression: e,
            op,
            strength: strength.into(),
        }
    }
    /// The expression of the left hand side of the constraint equation.
    pub fn expr(&self) -> &Expression<T> {
        &self.expression
    }
    /// The relational operator governing the constraint.
    pub fn op(&self) -> RelationalOperator {
        self.op
    }
    /// The strength of the constraint that the solver will use.
    pub fn strength(&self) -> f64 {
        self.strength.into_inner()
    }
    /// Set the strength in builder-style
    pub fn with_strength(self, s: f64) -> Self {
        let mut c = self;
        c.strength = s.into();
        c
    }
}

/// This is part of the syntactic sugar used for specifying constraints. This enum should be used as part of a
/// constraint expression. See the module documentation for more information.
#[derive(Debug)]
pub enum WeightedRelation {
    /// `==`
    EQ(f64),
    /// `<=`
    LE(f64),
    /// `>=`
    GE(f64),
}
impl From<WeightedRelation> for (RelationalOperator, f64) {
    fn from(r: WeightedRelation) -> (RelationalOperator, f64) {
        use WeightedRelation::*;
        match r {
            EQ(s) => (RelationalOperator::Equal, s),
            LE(s) => (RelationalOperator::LessOrEqual, s),
            GE(s) => (RelationalOperator::GreaterOrEqual, s),
        }
    }
}

/// This is an intermediate type used in the syntactic sugar for specifying constraints. You should not use it
/// directly.
#[derive(Debug)]
pub struct PartialConstraint<T>(pub Expression<T>, pub WeightedRelation);

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum SymbolType {
    Invalid,
    External,
    Slack,
    Error,
    Dummy,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct Symbol(usize, SymbolType);

impl Symbol {
    /// Choose the subject for solving for the row.
    ///
    /// This method will choose the best subject for using as the solve
    /// target for the row. An invalid symbol will be returned if there
    /// is no valid target.
    ///
    /// The symbols are chosen according to the following precedence:
    ///
    /// 1) The first symbol representing an external variable.
    /// 2) A negative slack or error tag variable.
    ///
    /// If a subject cannot be found, an invalid symbol will be returned.
    fn choose_subject(row: &Row, tag: &Tag) -> Symbol {
        for s in row.cells.keys() {
            if s.type_() == SymbolType::External {
                return *s;
            }
        }
        if (tag.marker.type_() == SymbolType::Slack || tag.marker.type_() == SymbolType::Error)
            && row.coefficient_for(tag.marker) < 0.0
        {
            return tag.marker;
        }
        if (tag.other.type_() == SymbolType::Slack || tag.other.type_() == SymbolType::Error)
            && row.coefficient_for(tag.other) < 0.0
        {
            return tag.other;
        }
        Symbol::invalid()
    }

    fn invalid() -> Symbol {
        Symbol(0, SymbolType::Invalid)
    }
    fn type_(&self) -> SymbolType {
        self.1
    }
}

#[derive(Copy, Clone, Debug)]
struct Tag {
    marker: Symbol,
    other: Symbol,
}

#[derive(Clone, Debug)]
struct Row {
    cells: FxHashMap<Symbol, OrderedFloat<f64>>,
    constant: OrderedFloat<f64>,
}

fn near_zero(value: f64) -> bool {
    const EPS: f64 = 1E-8;
    if value < 0.0 {
        -value < EPS
    } else {
        value < EPS
    }
}

impl Row {
    pub fn new(constant: f64) -> Row {
        Row {
            cells: FxHashMap::default(),
            constant: constant.into(),
        }
    }
    fn add(&mut self, v: f64) -> f64 {
        *(self.constant.as_mut()) += v;
        self.constant.into_inner()
    }
    fn insert_symbol(&mut self, s: Symbol, coefficient: f64) {
        match self.cells.entry(s) {
            Entry::Vacant(entry) => {
                if !near_zero(coefficient) {
                    entry.insert(coefficient.into());
                }
            }
            Entry::Occupied(mut entry) => {
                let ofloat = entry.get_mut();
                let float = ofloat.as_mut();
                *float += coefficient;
                if near_zero(*float) {
                    entry.remove();
                }
            }
        }
    }

    fn insert_row(&mut self, other: &Row, coefficient: f64) -> bool {
        let constant_diff = other.constant.as_ref() * coefficient;
        *self.constant.as_mut() += constant_diff;
        for (s, v) in &other.cells {
            self.insert_symbol(*s, v.into_inner() * coefficient);
        }
        constant_diff != 0.0
    }

    fn remove(&mut self, s: Symbol) {
        self.cells.remove(&s);
    }

    fn reverse_sign(&mut self) {
        *self.constant.as_mut() *= -1.0;
        for v in &mut self.cells.values_mut() {
            *v.as_mut() *= -1.0;
        }
    }

    fn solve_for_symbol(&mut self, s: Symbol) {
        let coeff = -1.0
            / match self.cells.entry(s) {
                Entry::Occupied(entry) => entry.remove().into_inner(),
                Entry::Vacant(_) => unreachable!(),
            };
        *self.constant.as_mut() *= coeff;
        for v in &mut self.cells.values_mut() {
            *v.as_mut() *= coeff;
        }
    }

    fn solve_for_symbols(&mut self, lhs: Symbol, rhs: Symbol) {
        self.insert_symbol(lhs, -1.0);
        self.solve_for_symbol(rhs);
    }

    fn coefficient_for(&self, s: Symbol) -> f64 {
        self.cells
            .get(&s)
            .cloned()
            .map(|o| o.into_inner())
            .unwrap_or(0.0)
    }

    fn substitute(&mut self, s: Symbol, row: &Row) -> bool {
        if let Some(coeff) = self.cells.remove(&s) {
            self.insert_row(row, coeff.into())
        } else {
            false
        }
    }

    /// Test whether a row is composed of all dummy variables.
    fn all_dummies(&self) -> bool {
        for symbol in self.cells.keys() {
            if symbol.type_() != SymbolType::Dummy {
                return false;
            }
        }
        true
    }

    /// Get the first Slack or Error symbol in the row.
    ///
    /// If no such symbol is present, and Invalid symbol will be returned.
    /// Never returns an External symbol
    fn any_pivotable_symbol(&self) -> Symbol {
        for symbol in self.cells.keys() {
            if symbol.type_() == SymbolType::Slack || symbol.type_() == SymbolType::Error {
                return *symbol;
            }
        }
        Symbol::invalid()
    }

    /// Compute the entering variable for a pivot operation.
    ///
    /// This method will return first symbol in the objective function which
    /// is non-dummy and has a coefficient less than zero. If no symbol meets
    /// the criteria, it means the objective function is at a minimum, and an
    /// invalid symbol is returned.
    /// Could return an External symbol
    fn get_entering_symbol(&self) -> Symbol {
        for (symbol, value) in &self.cells {
            if symbol.type_() != SymbolType::Dummy && *value.as_ref() < 0.0 {
                return *symbol;
            }
        }
        Symbol::invalid()
    }
}

/// The possible error conditions that `Solver::commit_edit` can fail with.
#[derive(Debug, Copy, Clone)]
pub struct EditConstraintError(&'static str);

/// The possible error conditions that `Solver::add_constraint` can fail with.
#[derive(Debug, Copy, Clone)]
pub enum AddConstraintError {
    /// The constraint specified has already been added to the solver.
    DuplicateConstraint,
    /// The constraint is required, but it is unsatisfiable in conjunction with the existing constraints.
    UnsatisfiableConstraint,
    /// The solver entered an invalid state. If this occurs please report the issue. This variant specifies
    /// additional details as a string.
    InternalSolverError(&'static str),
}

/// The possible error conditions that `Solver::remove_constraint` can fail with.
#[derive(Debug, Copy, Clone)]
pub enum RemoveConstraintError {
    /// The constraint specified was not already in the solver, so cannot be removed.
    UnknownConstraint,
    /// The solver entered an invalid state. If this occurs please report the issue. This variant specifies
    /// additional details as a string.
    InternalSolverError(&'static str),
}

/// The possible error conditions that `Solver::add_edit_variable` can fail with.
#[derive(Debug, Copy, Clone)]
pub enum AddEditVariableError {
    /// The specified variable is already marked as an edit variable in the solver.
    DuplicateEditVariable,
    /// The specified strength was `REQUIRED`. This is illegal for edit variable strengths.
    BadRequiredStrength,
}

/// The possible error conditions that `Solver::remove_edit_variable` can fail with.
#[derive(Debug, Copy, Clone)]
pub enum RemoveEditVariableError {
    /// The specified variable was not an edit variable in the solver, so cannot be removed.
    UnknownEditVariable,
    /// The solver entered an invalid state. If this occurs please report the issue. This variant specifies
    /// additional details as a string.
    InternalSolverError(&'static str),
}

/// The possible error conditions that `Solver::suggest_value` can fail with.
#[derive(Debug, Copy, Clone)]
pub enum SuggestValueError {
    /// The specified variable was not an edit variable in the solver, so cannot have its value suggested.
    UnknownEditVariable,
    /// The solver entered an invalid state. If this occurs please report the issue. This variant specifies
    /// additional details as a string.
    InternalSolverError(&'static str),
}

#[derive(Debug, Copy, Clone)]
pub struct InternalSolverError(&'static str);

pub use solver_impl::Solver;

#[cfg(test)]
mod tests {
    use super::*;
    use crate as casuarius;
    use std::{
        collections::HashMap,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static NEXT_K: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    pub struct Var(usize);
    derive_syntax_for!(Var);

    impl Var {
        pub fn new() -> Var {
            Var(NEXT_K.fetch_add(1, Ordering::Relaxed))
        }
    }

    #[test]
    fn example() {
        let mut names = HashMap::new();
        fn print_changes(names: &HashMap<Var, &'static str>, changes: &[(Var, f64)]) {
            println!("Changes:");
            for &(ref var, ref val) in changes {
                println!("{}: {}", names[var], val);
            }
        }

        let window_width = Var::new();
        names.insert(window_width, "window_width");
        struct Element {
            left: Var,
            right: Var,
        }
        let box1 = Element {
            left: Var::new(),
            right: Var::new(),
        };
        names.insert(box1.left, "box1.left");
        names.insert(box1.right, "box1.right");
        let box2 = Element {
            left: Var::new(),
            right: Var::new(),
        };
        names.insert(box2.left, "box2.left");
        names.insert(box2.right, "box2.right");
        let mut solver = Solver::default();

        solver
            .add_constraint(window_width.is_ge(0.0))
            .expect("Could not add window width >= 0");
        solver
            .add_constraint(window_width.is_le(1000.0))
            .expect("Could not add window width <= 1000.0");
        solver
            .add_constraint(box1.left.is(0.0))
            .expect("Could not add left align constraint");
        solver
            .add_constraint(box2.right.is(window_width))
            .expect("Could not add right align constraint");
        solver
            .add_constraint(box2.left.is_ge(box1.right))
            .expect("Could not add no overlap constraint");

        solver
            .add_constraint(box1.right.is(box1.left + 50.0).with_strength(WEAK))
            .expect("Could not add box1 width constraint");
        solver
            .add_constraint(box2.right.is(box2.left + 100.0).with_strength(WEAK))
            .expect("Could not add box2 width constraint");

        solver
            .add_constraint(box1.left.is_le(box1.right))
            .expect("Could not add box1 positive width constraint");
        solver
            .add_constraint(box2.left.is_le(box2.right))
            .expect("Could not add box2 positive width constraint");

        solver
            .add_edit_variable(window_width, STRONG)
            .expect("Could not add window width edit var");
        solver
            .suggest_value(window_width, 1000.0)
            .expect("Could not suggest window width = 1000");
        print_changes(&names, solver.fetch_changes());

        solver
            .suggest_value(window_width, 75.0)
            .expect("Could not suggest window width = 75");
        print_changes(&names, solver.fetch_changes());
        solver
            .add_constraint(
                ((box1.right - box1.left) / 50.0f64)
                    .is((box2.right - box2.left) / 100.0)
                    .with_strength(MEDIUM),
            )
            .unwrap();
        print_changes(&names, solver.fetch_changes());
    }

    #[test]
    fn test_quadrilateral() {
        struct Point {
            x: Var,
            y: Var,
        }
        impl Point {
            fn new() -> Point {
                Point {
                    x: Var::new(),
                    y: Var::new(),
                }
            }
        }

        let points = [Point::new(), Point::new(), Point::new(), Point::new()];
        let point_starts = [(10.0, 10.0), (10.0, 200.0), (200.0, 200.0), (200.0, 10.0)];
        let midpoints = [Point::new(), Point::new(), Point::new(), Point::new()];
        let mut solver = Solver::default();
        let mut weight = 1.0;
        let multiplier = 2.0;
        solver.begin_edit();
        for i in 0..4 {
            solver
                .add_constraints(vec![
                    (points[i].x)
                        .is(point_starts[i].0)
                        .with_strength(WEAK * weight),
                    //points[i].x | EQ(WEAK * weight) | point_starts[i].0,
                    (points[i].y)
                        .is(point_starts[i].1)
                        .with_strength(WEAK * weight),
                    //points[i].y | EQ(WEAK * weight) | point_starts[i].1,
                ])
                .expect("Could not add initial quad points");
            weight *= multiplier;
        }

        for (start, end) in vec![(0, 1), (1, 2), (2, 3), (3, 0)] {
            solver
                .add_constraints(vec![
                    (midpoints[start].x).is((points[start].x + points[end].x) / 2.0),
                    //midpoints[start].x | EQ(REQUIRED) | (points[start].x + points[end].x) / 2.0,
                    (midpoints[start].y).is((points[start].y + points[end].y) / 2.0),
                    //midpoints[start].y | EQ(REQUIRED) | (points[start].y + points[end].y) / 2.0,
                ])
                .expect("Could not add quad midpoints");
        }

        solver
            .add_constraints(vec![
                (points[0].x + 20.0f64).is_le(points[2].x),
                (points[0].x + 20.0f64).is_le(points[3].x),
                (points[1].x + 20.0f64).is_le(points[2].x),
                (points[1].x + 20.0f64).is_le(points[3].x),
                (points[0].y + 20.0f64).is_le(points[1].y),
                (points[0].y + 20.0f64).is_le(points[2].y),
                (points[3].y + 20.0f64).is_le(points[1].y),
                (points[3].y + 20.0f64).is_le(points[2].y),
            ])
            .expect("Could not add quad midpoint constraints");

        for point in &points {
            solver
                .add_constraints(vec![
                    point.x.is_ge(0.0),
                    point.y.is_ge(0.0),
                    point.x.is_le(500.0),
                    point.y.is_le(500.0),
                ])
                .expect("Could not add required bounds on quad");
        }
        solver
            .commit_edit()
            .expect("Could not commit constraint edit");

        assert_eq!(
            [
                (
                    solver.get_value(midpoints[0].x),
                    solver.get_value(midpoints[0].y)
                ),
                (
                    solver.get_value(midpoints[1].x),
                    solver.get_value(midpoints[1].y)
                ),
                (
                    solver.get_value(midpoints[2].x),
                    solver.get_value(midpoints[2].y)
                ),
                (
                    solver.get_value(midpoints[3].x),
                    solver.get_value(midpoints[3].y)
                )
            ],
            [(10.0, 105.0), (105.0, 200.0), (200.0, 105.0), (105.0, 10.0)]
        );

        solver
            .add_edit_variable(points[2].x, STRONG)
            .expect("Could not add x edit variable for 2nd point");
        solver
            .add_edit_variable(points[2].y, STRONG)
            .expect("Could not add y edit variable for 2nd point");
        solver
            .suggest_value(points[2].x, 300.0)
            .expect("Could not suggest value for x edit variable for 2nd point");
        solver
            .suggest_value(points[2].y, 400.0)
            .expect("Could not suggest value for y edit variable for 2nd point");

        assert_eq!(
            [
                (solver.get_value(points[0].x), solver.get_value(points[0].y)),
                (solver.get_value(points[1].x), solver.get_value(points[1].y)),
                (solver.get_value(points[2].x), solver.get_value(points[2].y)),
                (solver.get_value(points[3].x), solver.get_value(points[3].y))
            ],
            [(10.0, 10.0), (10.0, 200.0), (300.0, 400.0), (200.0, 10.0)]
        );

        assert_eq!(
            [
                (
                    solver.get_value(midpoints[0].x),
                    solver.get_value(midpoints[0].y)
                ),
                (
                    solver.get_value(midpoints[1].x),
                    solver.get_value(midpoints[1].y)
                ),
                (
                    solver.get_value(midpoints[2].x),
                    solver.get_value(midpoints[2].y)
                ),
                (
                    solver.get_value(midpoints[3].x),
                    solver.get_value(midpoints[3].y)
                )
            ],
            [(10.0, 105.0), (155.0, 300.0), (250.0, 205.0), (105.0, 10.0)]
        );
    }

    #[test]
    fn can_add_and_remove_constraints() {
        let mut solver = Solver::default();

        let var = Var(0);

        let constraint: Constraint<Var> = var.is(100.0);
        solver.add_constraint(constraint.clone()).unwrap();
        assert_eq!(solver.get_value(var), 100.0);

        solver.remove_constraint(&constraint).unwrap();
        solver.add_constraint(var.is(0.0)).unwrap();
        assert_eq!(solver.get_value(var), 0.0);
    }

    #[test]
    fn lib_doctest_part_one() {
        struct Element {
            left: Variable,
            right: Variable,
        }
        let box1 = Element {
            left: Variable("box1.left"),
            right: Variable("box1.right"),
        };

        let window_width = Variable("window_width");

        let box2 = Element {
            left: Variable("box2.left"),
            right: Variable("box2.right"),
        };

        let mut solver = Solver::<Variable>::default();
        solver
            .add_constraints(vec![
                window_width.is_ge(0.0),     // positive window width
                box1.left.is(0.0),           // left align
                box2.right.is(window_width), // right align
                box2.left.is_ge(box1.right), // no overlap
                // preferred widths:
                (box1.right - box1.left).is(50.0),
                (box2.right - box2.left).is(100.0),
                // positive widths
                box1.left.is_le(box1.right).with_strength(WEAK),
                box2.left.is_le(box2.right).with_strength(WEAK),
            ])
            .unwrap();

        solver.add_edit_variable(window_width, STRONG).unwrap();
        solver.suggest_value(window_width, 300.0).unwrap();

        let mut print_changes = || {
            println!("Changes:");
            solver
                .fetch_changes()
                .iter()
                .for_each(|(var, val)| println!("{}: {}", var.0, val));
        };
        print_changes();

        let ww = solver.get_value(window_width);
        let b1l = solver.get_value(box1.left);
        let b1r = solver.get_value(box1.right);
        let b2l = solver.get_value(box2.left);
        let b2r = solver.get_value(box2.right);
        println!("window_width: {}", ww);
        println!("box1.left {}", b1l);
        println!("box1.right {}", b1r);
        println!("box2.left {}", b2l);
        println!("box2.right {}", b2r);
        assert!(ww >= 0.0, "window_width >= 0.0");
        assert_eq!(0.0, b1l, "box1.left ({}) == 0", b1l);
        assert_eq!(ww, b2r, "box2.right ({}) != ww ({})", b2r, ww);
        assert!(b2l >= b1r, "box2.left >= box1.right");
        assert!(b1l <= b1r, "box1.left <= box1.right");
        assert!(b2l <= b2r, "box2.left <= box2.right");
        assert_eq!(50.0, b1r - b1l, "box1 width");
        assert_eq!(100.0, b2r - b2l, "box2 width");
    }

    #[test]
    fn neg_zero_sanity() {
        let nzero: f64 = -0.0;
        let zero: f64 = 0.0;
        assert!(nzero == zero);
        assert!(!(nzero < zero));
        assert!(nzero >= zero);
    }
}
