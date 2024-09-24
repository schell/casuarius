use crate::{Constraint, Expression, Term};

/// A trait for creating constraints using custom variable types.
pub trait Constrainable<Var>
where
    Var: Sized,
    Self: Sized,
{
    fn equal_to<X>(self, x: X) -> Constraint<Var>
    where
        X: Into<Expression<Var>> + Clone;

    fn is<X>(self, x: X) -> Constraint<Var>
    where
        X: Into<Expression<Var>> + Clone,
    {
        self.equal_to(x)
    }

    fn greater_than_or_equal_to<X>(self, x: X) -> Constraint<Var>
    where
        X: Into<Expression<Var>> + Clone;

    fn is_ge<X>(self, x: X) -> Constraint<Var>
    where
        X: Into<Expression<Var>> + Clone,
    {
        self.greater_than_or_equal_to(x)
    }

    fn less_than_or_equal_to<X>(self, x: X) -> Constraint<Var>
    where
        X: Into<Expression<Var>> + Clone;

    fn is_le<X>(self, x: X) -> Constraint<Var>
    where
        X: Into<Expression<Var>> + Clone,
    {
        self.less_than_or_equal_to(x)
    }
}

// Term

impl<T> std::ops::Mul<f64> for Term<T> {
    type Output = Term<T>;
    fn mul(mut self, v: f64) -> Term<T> {
        self *= v;
        self
    }
}

impl<T> std::ops::MulAssign<f64> for Term<T> {
    fn mul_assign(&mut self, v: f64) {
        *(self.coefficient.as_mut()) *= v;
    }
}

impl<T> std::ops::Mul<f32> for Term<T> {
    type Output = Term<T>;
    fn mul(self, v: f32) -> Term<T> {
        self.mul(v as f64)
    }
}

impl<T> std::ops::MulAssign<f32> for Term<T> {
    fn mul_assign(&mut self, v: f32) {
        self.mul_assign(v as f64)
    }
}

impl<T> std::ops::Mul<Term<T>> for f64 {
    type Output = Term<T>;
    fn mul(self, mut t: Term<T>) -> Term<T> {
        *(t.coefficient.as_mut()) *= self;
        t
    }
}

impl<T> std::ops::Mul<Term<T>> for f32 {
    type Output = Term<T>;
    fn mul(self, t: Term<T>) -> Term<T> {
        (self as f64).mul(t)
    }
}

impl<T> std::ops::Div<f64> for Term<T> {
    type Output = Term<T>;
    fn div(mut self, v: f64) -> Term<T> {
        self /= v;
        self
    }
}

impl<T> std::ops::DivAssign<f64> for Term<T> {
    fn div_assign(&mut self, v: f64) {
        *(self.coefficient.as_mut()) /= v;
    }
}

impl<T> std::ops::Div<f32> for Term<T> {
    type Output = Term<T>;
    fn div(self, v: f32) -> Term<T> {
        self.div(v as f64)
    }
}

impl<T> std::ops::DivAssign<f32> for Term<T> {
    fn div_assign(&mut self, v: f32) {
        self.div_assign(v as f64)
    }
}

impl<T: Clone> std::ops::Add<f64> for Term<T> {
    type Output = Expression<T>;
    fn add(self, v: f64) -> Expression<T> {
        Expression::new(vec![self], v)
    }
}

impl<T: Clone> std::ops::Add<f32> for Term<T> {
    type Output = Expression<T>;
    fn add(self, v: f32) -> Expression<T> {
        self.add(v as f64)
    }
}

impl<T: Clone> std::ops::Add<Term<T>> for f64 {
    type Output = Expression<T>;
    fn add(self, t: Term<T>) -> Expression<T> {
        Expression::new(vec![t], self)
    }
}

impl<T: Clone> std::ops::Add<Term<T>> for f32 {
    type Output = Expression<T>;
    fn add(self, t: Term<T>) -> Expression<T> {
        (self as f64).add(t)
    }
}

impl<T: Clone> std::ops::Add<Term<T>> for Term<T> {
    type Output = Expression<T>;
    fn add(self, t: Term<T>) -> Expression<T> {
        Expression::new(vec![self, t], 0.0)
    }
}

impl<T> std::ops::Add<Expression<T>> for Term<T> {
    type Output = Expression<T>;
    fn add(self, mut e: Expression<T>) -> Expression<T> {
        e.terms.push(self);
        e
    }
}

impl<T> std::ops::Add<Term<T>> for Expression<T> {
    type Output = Expression<T>;
    fn add(mut self, t: Term<T>) -> Expression<T> {
        self += t;
        self
    }
}

impl<T> std::ops::AddAssign<Term<T>> for Expression<T> {
    fn add_assign(&mut self, t: Term<T>) {
        self.terms.push(t);
    }
}

impl<T> std::ops::Neg for Term<T> {
    type Output = Term<T>;
    fn neg(mut self) -> Term<T> {
        *(self.coefficient.as_mut()) = -(self.coefficient.into_inner());
        self
    }
}

impl<T: Clone> std::ops::Sub<f64> for Term<T> {
    type Output = Expression<T>;
    fn sub(self, v: f64) -> Expression<T> {
        Expression::new(vec![self], -v)
    }
}

impl<T: Clone> std::ops::Sub<f32> for Term<T> {
    type Output = Expression<T>;
    fn sub(self, v: f32) -> Expression<T> {
        self.sub(v as f64)
    }
}

impl<T: Clone> std::ops::Sub<Term<T>> for f64 {
    type Output = Expression<T>;
    fn sub(self, t: Term<T>) -> Expression<T> {
        Expression::new(vec![-t], self)
    }
}

impl<T: Clone> std::ops::Sub<Term<T>> for f32 {
    type Output = Expression<T>;
    fn sub(self, t: Term<T>) -> Expression<T> {
        (self as f64).sub(t)
    }
}

impl<T: Clone> std::ops::Sub<Term<T>> for Term<T> {
    type Output = Expression<T>;
    fn sub(self, t: Term<T>) -> Expression<T> {
        Expression::new(vec![self, -t], 0.0)
    }
}

impl<T: Clone> std::ops::Sub<Expression<T>> for Term<T> {
    type Output = Expression<T>;
    fn sub(self, mut e: Expression<T>) -> Expression<T> {
        e.negate();
        e.terms.push(self);
        e
    }
}

impl<T> std::ops::Sub<Term<T>> for Expression<T> {
    type Output = Expression<T>;
    fn sub(mut self, t: Term<T>) -> Expression<T> {
        self -= t;
        self
    }
}

impl<T> std::ops::SubAssign<Term<T>> for Expression<T> {
    fn sub_assign(&mut self, t: Term<T>) {
        self.terms.push(-t);
    }
}

// Expression

impl<T: Clone> std::ops::Mul<f64> for Expression<T> {
    type Output = Expression<T>;
    fn mul(mut self, v: f64) -> Expression<T> {
        self *= v.clone();
        self
    }
}

impl<T: Clone> std::ops::MulAssign<f64> for Expression<T> {
    fn mul_assign(&mut self, v: f64) {
        *(self.constant.as_mut()) *= v;
        for t in &mut self.terms {
            *t = t.clone() * v;
        }
    }
}

impl<T: Clone> std::ops::Mul<Expression<T>> for f64 {
    type Output = Expression<T>;
    fn mul(self, mut e: Expression<T>) -> Expression<T> {
        *(e.constant.as_mut()) *= self;
        for t in &mut e.terms {
            *t = t.clone() * self;
        }
        e
    }
}

impl<T: Clone> std::ops::Div<f64> for Expression<T> {
    type Output = Expression<T>;
    fn div(mut self, v: f64) -> Expression<T> {
        self /= v;
        self
    }
}

impl<T: Clone> std::ops::DivAssign<f64> for Expression<T> {
    fn div_assign(&mut self, v: f64) {
        *(self.constant.as_mut()) /= v;
        for t in &mut self.terms {
            *t = t.clone() / v;
        }
    }
}

impl<T> std::ops::Add<f64> for Expression<T> {
    type Output = Expression<T>;
    fn add(mut self, v: f64) -> Expression<T> {
        self += v;
        self
    }
}

impl<T> std::ops::AddAssign<f64> for Expression<T> {
    fn add_assign(&mut self, v: f64) {
        *(self.constant.as_mut()) += v;
    }
}

impl<T> std::ops::Add<Expression<T>> for f64 {
    type Output = Expression<T>;
    fn add(self, mut e: Expression<T>) -> Expression<T> {
        *(e.constant.as_mut()) += self;
        e
    }
}

impl<T> std::ops::Add<Expression<T>> for Expression<T> {
    type Output = Expression<T>;
    fn add(mut self, e: Expression<T>) -> Expression<T> {
        self += e;
        self
    }
}

impl<T> std::ops::AddAssign<Expression<T>> for Expression<T> {
    fn add_assign(&mut self, mut e: Expression<T>) {
        self.terms.append(&mut e.terms);
        *(self.constant.as_mut()) += e.constant.into_inner();
    }
}

impl<T: Clone> std::ops::Neg for Expression<T> {
    type Output = Expression<T>;
    fn neg(mut self) -> Expression<T> {
        self.negate();
        self
    }
}

impl<T> std::ops::Sub<f64> for Expression<T> {
    type Output = Expression<T>;
    fn sub(mut self, v: f64) -> Expression<T> {
        self -= v;
        self
    }
}

impl<T> std::ops::SubAssign<f64> for Expression<T> {
    fn sub_assign(&mut self, v: f64) {
        *(self.constant.as_mut()) -= v;
    }
}

impl<T: Clone> std::ops::Sub<Expression<T>> for f64 {
    type Output = Expression<T>;
    fn sub(self, mut e: Expression<T>) -> Expression<T> {
        e.negate();
        *(e.constant.as_mut()) += self;
        e
    }
}

impl<T: Clone> std::ops::Sub<Expression<T>> for Expression<T> {
    type Output = Expression<T>;
    fn sub(mut self, e: Expression<T>) -> Expression<T> {
        self -= e;
        self
    }
}

impl<T: Clone> std::ops::SubAssign<Expression<T>> for Expression<T> {
    fn sub_assign(&mut self, mut e: Expression<T>) {
        e.negate();
        self.terms.append(&mut e.terms);
        *(self.constant.as_mut()) += e.constant.into_inner();
    }
}

macro_rules! derive_expr_ops_for {
    ( $x:ty ) => {
        impl<T: Clone> std::ops::Mul<$x> for Expression<T> {
            type Output = Expression<T>;
            fn mul(self, v: $x) -> Expression<T> {
                self.mul(v as f64)
            }
        }

        impl<T: Clone> std::ops::MulAssign<$x> for Expression<T> {
            fn mul_assign(&mut self, v: $x) {
                let v2 = v as f64;
                *(self.constant.as_mut()) *= v2;
                for t in &mut self.terms {
                    *t = t.clone() * v2;
                }
            }
        }

        impl<T: Clone> std::ops::Mul<Expression<T>> for $x {
            type Output = Expression<T>;
            fn mul(self, e: Expression<T>) -> Expression<T> {
                (self as f64).mul(e)
            }
        }

        impl<T: Clone> std::ops::Div<$x> for Expression<T> {
            type Output = Expression<T>;
            fn div(self, v: $x) -> Expression<T> {
                self.div(v as f64)
            }
        }

        impl<T: Clone> std::ops::DivAssign<$x> for Expression<T> {
            fn div_assign(&mut self, v: $x) {
                self.div_assign(v as f64)
            }
        }

        impl<T> std::ops::Add<$x> for Expression<T> {
            type Output = Expression<T>;
            fn add(self, v: $x) -> Expression<T> {
                self.add(v as f64)
            }
        }

        impl<T> std::ops::AddAssign<$x> for Expression<T> {
            fn add_assign(&mut self, v: $x) {
                self.add_assign(v as f64)
            }
        }

        impl<T> std::ops::Add<Expression<T>> for $x {
            type Output = Expression<T>;
            fn add(self, e: Expression<T>) -> Expression<T> {
                (self as f64).add(e)
            }
        }

        impl<T> std::ops::Sub<$x> for Expression<T> {
            type Output = Expression<T>;
            fn sub(self, v: $x) -> Expression<T> {
                self.sub(v as f64)
            }
        }

        impl<T: Clone> std::ops::Sub<Expression<T>> for $x {
            type Output = Expression<T>;
            fn sub(self, e: Expression<T>) -> Expression<T> {
                (self as f64).sub(e)
            }
        }

        impl<T> std::ops::SubAssign<$x> for Expression<T> {
            fn sub_assign(&mut self, v: $x) {
                self.sub_assign(v as f64)
            }
        }
    };
}

derive_expr_ops_for!(f32);
derive_expr_ops_for!(i32);
derive_expr_ops_for!(u32);
