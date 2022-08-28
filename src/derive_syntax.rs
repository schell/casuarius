/// Derives operator support for your cassowary solver variable type.
/// This allows you to use your variable type in writing expressions, to a limited extent.
#[macro_export]
macro_rules! derive_syntax_for {
    ( $x:ty ) => {
        impl From<$x> for casuarius::Expression<$x> {
            fn from(v: $x) -> casuarius::Expression<$x> {
                casuarius::Expression::from_term(casuarius::Term::new(v, 1.0))
            }
        }

        impl std::ops::Add<f64> for $x {
            type Output = casuarius::Expression<$x>;
            fn add(self, v: f64) -> casuarius::Expression<$x> {
                casuarius::Expression::new(vec![casuarius::Term::new(self, 1.0)], v)
            }
        }

        impl std::ops::Add<f32> for $x {
            type Output = casuarius::Expression<$x>;
            fn add(self, v: f32) -> casuarius::Expression<$x> {
                self.add(v as f64)
            }
        }

        impl std::ops::Add<u32> for $x {
            type Output = casuarius::Expression<$x>;
            fn add(self, v: u32) -> casuarius::Expression<$x> {
                self.add(v as f64)
            }
        }

        impl std::ops::Add<$x> for f64 {
            type Output = casuarius::Expression<$x>;
            fn add(self, v: $x) -> casuarius::Expression<$x> {
                casuarius::Expression::new(vec![casuarius::Term::new(v, 1.0)], self)
            }
        }

        impl std::ops::Add<$x> for f32 {
            type Output = casuarius::Expression<$x>;
            fn add(self, v: $x) -> casuarius::Expression<$x> {
                (self as f64).add(v)
            }
        }

        impl std::ops::Add<$x> for u32 {
            type Output = casuarius::Expression<$x>;
            fn add(self, v: $x) -> casuarius::Expression<$x> {
                (self as f64).add(v)
            }
        }

        impl std::ops::Add<$x> for $x {
            type Output = casuarius::Expression<$x>;
            fn add(self, v: $x) -> casuarius::Expression<$x> {
                casuarius::Expression::new(
                    vec![
                        casuarius::Term::new(self, 1.0),
                        casuarius::Term::new(v, 1.0),
                    ],
                    0.0,
                )
            }
        }

        impl std::ops::Add<casuarius::Term<$x>> for $x {
            type Output = casuarius::Expression<$x>;
            fn add(self, t: casuarius::Term<$x>) -> casuarius::Expression<$x> {
                casuarius::Expression::new(vec![casuarius::Term::new(self, 1.0), t], 0.0)
            }
        }

        impl std::ops::Add<$x> for casuarius::Term<$x> {
            type Output = casuarius::Expression<$x>;
            fn add(self, v: $x) -> casuarius::Expression<$x> {
                casuarius::Expression::new(vec![self, casuarius::Term::new(v, 1.0)], 0.0)
            }
        }

        impl std::ops::Add<casuarius::Expression<$x>> for $x {
            type Output = casuarius::Expression<$x>;
            fn add(self, mut e: casuarius::Expression<$x>) -> casuarius::Expression<$x> {
                e.terms.push(casuarius::Term::new(self, 1.0));
                e
            }
        }

        impl std::ops::Add<$x> for casuarius::Expression<$x> {
            type Output = casuarius::Expression<$x>;
            fn add(mut self, v: $x) -> casuarius::Expression<$x> {
                self += v;
                self
            }
        }

        impl std::ops::AddAssign<$x> for casuarius::Expression<$x> {
            fn add_assign(&mut self, v: $x) {
                self.terms.push(casuarius::Term::new(v, 1.0));
            }
        }

        impl std::ops::Neg for $x {
            type Output = casuarius::Term<$x>;
            fn neg(self) -> casuarius::Term<$x> {
                casuarius::Term::new(self, -1.0)
            }
        }

        impl std::ops::Sub<f64> for $x {
            type Output = casuarius::Expression<$x>;
            fn sub(self, v: f64) -> casuarius::Expression<$x> {
                casuarius::Expression::new(vec![casuarius::Term::new(self, 1.0)], -v)
            }
        }

        impl std::ops::Sub<f32> for $x {
            type Output = casuarius::Expression<$x>;
            fn sub(self, v: f32) -> casuarius::Expression<$x> {
                self.sub(v as f64)
            }
        }

        impl std::ops::Sub<u32> for $x {
            type Output = casuarius::Expression<$x>;
            fn sub(self, v: u32) -> casuarius::Expression<$x> {
                self.sub(v as f64)
            }
        }

        impl std::ops::Sub<$x> for f64 {
            type Output = casuarius::Expression<$x>;
            fn sub(self, v: $x) -> casuarius::Expression<$x> {
                casuarius::Expression::new(vec![casuarius::Term::new(v, -1.0)], self)
            }
        }

        impl std::ops::Sub<$x> for f32 {
            type Output = casuarius::Expression<$x>;
            fn sub(self, v: $x) -> casuarius::Expression<$x> {
                (self as f64).sub(v)
            }
        }

        impl std::ops::Sub<$x> for u32 {
            type Output = casuarius::Expression<$x>;
            fn sub(self, v: $x) -> casuarius::Expression<$x> {
                (self as f64).sub(v)
            }
        }

        impl std::ops::Sub<$x> for $x {
            type Output = casuarius::Expression<$x>;
            fn sub(self, v: $x) -> casuarius::Expression<$x> {
                casuarius::Expression::new(
                    vec![
                        casuarius::Term::new(self, 1.0),
                        casuarius::Term::new(v, -1.0),
                    ],
                    0.0,
                )
            }
        }

        impl std::ops::Sub<casuarius::Term<$x>> for $x {
            type Output = casuarius::Expression<$x>;
            fn sub(self, t: casuarius::Term<$x>) -> casuarius::Expression<$x> {
                casuarius::Expression::new(vec![casuarius::Term::new(self, 1.0), -t], 0.0)
            }
        }

        impl std::ops::Sub<$x> for casuarius::Term<$x> {
            type Output = casuarius::Expression<$x>;
            fn sub(self, v: $x) -> casuarius::Expression<$x> {
                casuarius::Expression::new(vec![self, casuarius::Term::new(v, -1.0)], 0.0)
            }
        }

        impl std::ops::Sub<casuarius::Expression<$x>> for $x {
            type Output = casuarius::Expression<$x>;
            fn sub(self, mut e: casuarius::Expression<$x>) -> casuarius::Expression<$x> {
                e.negate();
                e.terms.push(casuarius::Term::new(self, 1.0));
                e
            }
        }

        impl std::ops::Sub<$x> for casuarius::Expression<$x> {
            type Output = casuarius::Expression<$x>;
            fn sub(mut self, v: $x) -> casuarius::Expression<$x> {
                self -= v;
                self
            }
        }

        impl std::ops::SubAssign<$x> for casuarius::Expression<$x> {
            fn sub_assign(&mut self, v: $x) {
                self.terms.push(casuarius::Term::new(v, -1.0));
            }
        }

        impl std::ops::Mul<f64> for $x {
            type Output = casuarius::Term<$x>;
            fn mul(self, v: f64) -> casuarius::Term<$x> {
                casuarius::Term::new(self, v)
            }
        }

        impl std::ops::Mul<f32> for $x {
            type Output = casuarius::Term<$x>;
            fn mul(self, v: f32) -> casuarius::Term<$x> {
                self.mul(v as f64)
            }
        }

        impl std::ops::Mul<$x> for f64 {
            type Output = casuarius::Term<$x>;
            fn mul(self, v: $x) -> casuarius::Term<$x> {
                casuarius::Term::new(v, self)
            }
        }

        impl std::ops::Mul<$x> for f32 {
            type Output = casuarius::Term<$x>;
            fn mul(self, v: $x) -> casuarius::Term<$x> {
                (self as f64).mul(v)
            }
        }

        impl std::ops::Div<f64> for $x {
            type Output = casuarius::Term<$x>;
            fn div(self, v: f64) -> casuarius::Term<$x> {
                casuarius::Term::new(self, 1.0 / v)
            }
        }

        impl std::ops::Div<f32> for $x {
            type Output = casuarius::Term<$x>;
            fn div(self, v: f32) -> casuarius::Term<$x> {
                self.div(v as f64)
            }
        }

        impl casuarius::Constrainable<$x> for $x {
            fn equal_to<X>(self, x: X) -> casuarius::Constraint<$x>
            where
                X: Into<casuarius::Expression<$x>> + Clone,
            {
                let lhs: casuarius::Expression<$x> = self.into();
                let rhs: casuarius::Expression<$x> = x.into();
                lhs.equal_to(rhs)
            }
            fn greater_than_or_equal_to<X>(self, x: X) -> casuarius::Constraint<$x>
            where
                X: Into<casuarius::Expression<$x>> + Clone,
            {
                let lhs: casuarius::Expression<$x> = self.into();
                let rhs: casuarius::Expression<$x> = x.into();
                lhs.is_ge(rhs)
            }
            fn less_than_or_equal_to<X>(self, x: X) -> casuarius::Constraint<$x>
            where
                X: Into<casuarius::Expression<$x>> + Clone,
            {
                let lhs: casuarius::Expression<$x> = self.into();
                let rhs: casuarius::Expression<$x> = x.into();
                lhs.is_le(rhs)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::{self as casuarius, Solver, Constrainable};

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum VariableX {
        Left(usize),
        Width(usize),
    }
    derive_syntax_for!(VariableX);

    struct Element(usize);

    impl Element {
        fn left(&self) -> VariableX {
            VariableX::Left(self.0)
        }
        fn width(&self) -> VariableX {
            VariableX::Width(self.0)
        }
    }

    #[test]
    fn can_do_ops() {
        let el0 = Element(0);
        let el1 = Element(1);

        let mut solver_x = Solver::default();
        solver_x
            .add_constraints(vec![
                el0.left().is(0.0),
                el0.width().is(100.0),
                el1.left().is_ge(el0.left() + el0.width()),
            ])
            .unwrap();
        assert_eq!(solver_x.get_value(el0.left()), 0.0);
        assert_eq!(solver_x.get_value(el0.width()), 100.0);
        assert_eq!(solver_x.get_value(el1.left()), 100.0);
    }
}
