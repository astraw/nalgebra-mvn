//! Multivariate normal distribution using [nalgebra](https://nalgebra.org).
//!
//! # Example of usage
//!
//! ```
//! use nalgebra::{ Vector2, Matrix2, OMatrix, U2, U3};
//! use nalgebra_mvn::MultivariateNormal;
//!
//! // specify mean and covariance of our multi-variate normal
//! let mu = Vector2::from_row_slice(&[9.0, 1.0]);
//! let sigma = Matrix2::from_row_slice(
//!     &[1.0, 0.0,
//!     0.0, 1.0]);
//!
//! let mvn = MultivariateNormal::from_mean_and_covariance(&mu, &sigma).unwrap();
//!
//! // input samples are row vectors vertically stacked
//! let xs = OMatrix::<_,U3,U2>::new(
//!     8.9, 1.0,
//!     9.0, 1.0,
//!     9.1, 1.0,
//! );
//!
//! // evaluate the density at each of our samples.
//! let result = mvn.pdf(&xs);
//!
//! // result is a vector with num samples rows
//! assert!(result.nrows()==xs.nrows());
//! ```
//!
//! # License
//! Licensed under either of
//!
//! * Apache License, Version 2.0,
//!   (./LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//! * MIT license (./LICENSE-MIT or http://opensource.org/licenses/MIT)
//!
//! at your option.

use nalgebra::{allocator::Allocator, linalg, DefaultAllocator, Dim, OMatrix, OVector, RealField};

/// An error
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
}

impl Error {
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.kind)
    }
}

/// Kind of error
#[derive(Debug, Clone)]
pub enum ErrorKind {
    NotDefinitePositive,
}

/// An `N`-dimensional multivariate normal distribution
///
/// See the [crate-level docs](index.html) for example usage.
#[derive(Debug, Clone)]
pub struct MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + nalgebra::DimMin<N, Output = N>,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, nalgebra::U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as nalgebra::DimMin<N>>::Output>,
{
    /// Negative of mean of the distribution
    neg_mu: nalgebra::OVector<Real, N>,
    /// Precision of the distribution (the inverse of covariance)
    precision: nalgebra::OMatrix<Real, N, N>,
    /// A cached value used for calculating the density
    fac: Real,
}

impl<Real, N> MultivariateNormal<Real, N>
where
    Real: RealField,
    N: Dim + nalgebra::DimMin<N, Output = N> + nalgebra::DimSub<nalgebra::Dynamic>,
    DefaultAllocator: Allocator<Real, N>,
    DefaultAllocator: Allocator<Real, N, N>,
    DefaultAllocator: Allocator<Real, nalgebra::U1, N>,
    DefaultAllocator: Allocator<(usize, usize), <N as nalgebra::DimMin<N>>::Output>,
{
    /// Create a multivariate normal distribution from a mean and precision
    ///
    /// The mean vector `mu` is N dimensional and the `precision` matrix is
    /// N x N.
    pub fn from_mean_and_precision(
        mu: &nalgebra::OVector<Real, N>,
        precision: &nalgebra::OMatrix<Real, N, N>,
    ) -> Self {
        // Here we calculate and cache `fac` to prevent computing it repeatedly.

        // The determinant of the inverse of an invertible matrix is
        // the inverse of the determinant.
        let precision_det = nalgebra::linalg::LU::new(precision.clone()).determinant();
        let det = Real::one() / precision_det;

        let ndim = mu.nrows();
        let fac: Real = Real::one() / (Real::two_pi().powi(ndim as i32) * det.abs()).sqrt();

        Self {
            neg_mu: -mu,
            precision: precision.clone(),
            fac,
        }
    }

    /// Create a multivariate normal distribution from a mean and covariance
    ///
    /// The mean vector `mu` is N dimensional and the `covariance` matrix is
    /// N x N.
    ///
    /// The precision matrix is calculated by inverting the covariance matrix
    /// using a Cholesky decomposition. This can fail if the covariance matrix
    /// is not definite positive.
    pub fn from_mean_and_covariance(
        mu: &nalgebra::OVector<Real, N>,
        covariance: &nalgebra::OMatrix<Real, N, N>,
    ) -> Result<Self, Error> {
        // calculate precision from covariance.
        let precision = linalg::Cholesky::new(covariance.clone())
            .ok_or(Error {
                kind: ErrorKind::NotDefinitePositive,
            })?
            .inverse();
        let result = Self::from_mean_and_precision(mu, &precision);
        Ok(result)
    }

    /// Get the mean of the distribution
    pub fn mean(&self) -> nalgebra::OVector<Real, N> {
        -&self.neg_mu
    }

    /// Get the precision of the distribution
    pub fn precision(&self) -> &nalgebra::OMatrix<Real, N, N> {
        &self.precision
    }

    fn inner_pdf<Count>(
        &self,
        xs_t: &nalgebra::OMatrix<Real, Count, N>,
    ) -> nalgebra::OVector<Real, Count>
    where
        Count: Dim,
        DefaultAllocator: Allocator<Real, Count>,
        DefaultAllocator: Allocator<Real, N, Count>,
        DefaultAllocator: Allocator<Real, Count, N>,
        DefaultAllocator: Allocator<Real, Count, Count>,
    {
        let dvs: nalgebra::OMatrix<Real, Count, N> = broadcast_add(&xs_t, &self.neg_mu);

        let left: nalgebra::OMatrix<Real, Count, N> = &dvs * &self.precision;
        let ny2_tmp: nalgebra::OMatrix<Real, Count, N> = left.component_mul(&dvs);
        let ones = nalgebra::OMatrix::<Real, N, nalgebra::U1>::repeat_generic(
            N::from_usize(self.neg_mu.nrows()),
            nalgebra::Const,
            nalgebra::convert::<f64, Real>(1.0),
        );
        let ny2: nalgebra::OVector<Real, Count> = ny2_tmp * ones;
        let y: nalgebra::OVector<Real, Count> = ny2 * nalgebra::convert::<f64, Real>(-0.5);
        y
    }

    /// Probability density function
    ///
    /// Evaluate the probability density at locations `xs`.
    pub fn pdf<Count>(
        &self,
        xs: &nalgebra::OMatrix<Real, Count, N>,
    ) -> nalgebra::OVector<Real, Count>
    where
        Count: Dim,
        DefaultAllocator: Allocator<Real, Count>,
        DefaultAllocator: Allocator<Real, N, Count>,
        DefaultAllocator: Allocator<Real, Count, N>,
        DefaultAllocator: Allocator<Real, Count, Count>,
    {
        let y = self.inner_pdf(xs);
        vec_exp(&y) * self.fac.clone()
    }

    /// Log of the probability density function
    ///
    /// Evaluate the log probability density at locations `xs`.
    pub fn logpdf<Count>(
        &self,
        xs: &nalgebra::OMatrix<Real, Count, N>,
    ) -> nalgebra::OVector<Real, Count>
    where
        Count: Dim,
        DefaultAllocator: Allocator<Real, Count>,
        DefaultAllocator: Allocator<Real, N, Count>,
        DefaultAllocator: Allocator<Real, Count, N>,
        DefaultAllocator: Allocator<Real, Count, Count>,
    {
        let y = self.inner_pdf(xs);
        vec_add(&y, self.fac.clone().ln())
    }
}

fn vec_exp<Real, Count>(v: &nalgebra::OVector<Real, Count>) -> nalgebra::OVector<Real, Count>
where
    Real: RealField,
    Count: Dim,
    DefaultAllocator: Allocator<Real, Count>,
{
    let nrows = Count::from_usize(v.nrows());
    OVector::from_iterator_generic(nrows, nalgebra::Const, v.iter().map(|vi| vi.clone().exp()))
}

fn vec_add<Real, Count>(
    v: &nalgebra::OVector<Real, Count>,
    rhs: Real,
) -> nalgebra::OVector<Real, Count>
where
    Real: RealField,
    Count: Dim,
    DefaultAllocator: Allocator<Real, Count>,
{
    let nrows = Count::from_usize(v.nrows());
    OVector::from_iterator_generic(
        nrows,
        nalgebra::Const,
        v.iter().map(|vi| vi.clone() + rhs.clone()),
    )
}

/// Add `vec` to each row of `arr`, returning the result with shape of `arr`.
///
/// Inputs `arr` has shape R x C and `vec` is C dimensional. Result
/// has shape R x C.
fn broadcast_add<Real, R, C>(
    arr: &OMatrix<Real, R, C>,
    vec: &OVector<Real, C>,
) -> OMatrix<Real, R, C>
where
    Real: RealField,
    R: Dim,
    C: Dim,
    DefaultAllocator: Allocator<Real, R, C>,
    DefaultAllocator: Allocator<Real, C>,
{
    let ndim = arr.nrows();
    let nrows = R::from_usize(arr.nrows());
    let ncols = C::from_usize(arr.ncols());

    // TODO: remove explicit index calculation and indexing
    OMatrix::from_iterator_generic(
        nrows,
        ncols,
        arr.iter().enumerate().map(|(i, el)| {
            let vi = i / ndim; // integer div to get index into vec
            el.clone() + vec[vi].clone()
        }),
    )
}

#[cfg(test)]
mod tests {
    use crate::*;
    use approx::assert_relative_eq;
    use nalgebra as na;

    /// Calculate the sample covariance
    ///
    /// Calculates the sample covariances among N-dimensional samples with M
    /// observations each. Calculates N x N covariance matrix from observations
    /// in `arr`, which is M rows of N columns used to store M vectors of
    /// dimension N.
    fn sample_covariance<Real: RealField, M: Dim, N: Dim>(
        arr: &OMatrix<Real, M, N>,
    ) -> nalgebra::OMatrix<Real, N, N>
    where
        DefaultAllocator: Allocator<Real, M, N>,
        DefaultAllocator: Allocator<Real, N, M>,
        DefaultAllocator: Allocator<Real, N, N>,
        DefaultAllocator: Allocator<Real, N>,
    {
        let mu: OVector<Real, N> = mean_axis0(arr);
        let y = broadcast_add(arr, &-mu);
        let n: Real = nalgebra::convert(arr.nrows() as f64);
        let sigma = (y.transpose() * y) / (n - Real::one());
        sigma
    }

    /// Calculate the mean of R x C matrix along the rows and return C dim vector
    fn mean_axis0<Real, R, C>(arr: &OMatrix<Real, R, C>) -> OVector<Real, C>
    where
        Real: RealField,
        R: Dim,
        C: Dim,
        DefaultAllocator: Allocator<Real, R, C>,
        DefaultAllocator: Allocator<Real, C>,
    {
        let vec_dim: C = C::from_usize(arr.ncols());
        let mut mu = OVector::<Real, C>::zeros_generic(vec_dim, nalgebra::Const);
        let scale: Real = Real::one() / na::convert(arr.nrows() as f64);
        for j in 0..arr.ncols() {
            let col_sum = arr
                .column(j)
                .iter()
                .fold(Real::zero(), |acc, x| acc + x.clone());
            mu[j] = col_sum * scale.clone();
        }
        mu
    }

    #[test]
    fn test_covar() {
        use nalgebra::core::dimension::{U2, U3};

        // We use the same example as https://numpy.org/doc/stable/reference/generated/numpy.cov.html

        // However, our format is transposed compared to numpy. We have
        // variables as the columns and samples as rows.
        let arr = OMatrix::<f64, U2, U3>::new(-2.1, -1.0, 4.3, 3.0, 1.1, 0.12).transpose();

        let c = sample_covariance(&arr);

        let expected = nalgebra::OMatrix::<f64, U2, U2>::new(11.71, -4.286, -4.286, 2.144133);

        assert_relative_eq!(c, expected, epsilon = 1e-3);
    }

    #[test]
    fn test_mean_and_precision() {
        let mu = na::Vector2::<f64>::new(0.0, 0.0);
        let precision = na::Matrix2::<f64>::new(1.0, 0.0, 0.0, 1.0);

        let mvn = MultivariateNormal::from_mean_and_precision(&mu, &precision);

        assert!(mu == mvn.mean());
        assert!(&precision == mvn.precision());
    }

    #[test]
    fn test_mean_axis0() {
        use nalgebra::core::dimension::{U2, U4};

        let a1 = OMatrix::<f64, U2, U4>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let actual1: OVector<f64, U4> = mean_axis0(&a1);
        let expected1 = &[3.0, 4.0, 5.0, 6.0];
        assert!(actual1.as_slice() == expected1);

        let a2 = OMatrix::<f64, U4, U2>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let actual2: OVector<f64, U2> = mean_axis0(&a2);
        let expected2 = &[4.0, 5.0];
        assert!(actual2.as_slice() == expected2);
    }

    #[test]
    fn test_broadcast_add() {
        use nalgebra::core::dimension::{U3, U4};

        let x = OMatrix::<f64, U3, U4>::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0, 200.0, 300.0, 400.0,
        );
        let v = OVector::<f64, U4>::new(-3.0, -4.0, -5.0, -3.0);
        let actual = broadcast_add(&x, &v);

        let expected = OMatrix::<f64, U3, U4>::new(
            -2.0, -2.0, -2.0, 1.0, 2.0, 2.0, 2.0, 5.0, 97.0, 196.0, 295.0, 397.0,
        );

        assert!(actual == expected);
    }

    #[test]
    fn test_density() {
        /*

        # This test in Python:

        from scipy.stats import multivariate_normal
        import numpy as np

        xs = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).T

        mu = np.array([
            [0],
            [0],
        ], dtype=np.float)

        covariance = np.array([
            [1, 0],
            [0, 1]], dtype=np.float)

        result = np.array(
            [0.15915494, 0.09653235, 0.09653235], dtype=np.float)

        result_py = multivariate_normal.pdf(xs, mean=mu[:,0], cov=covariance)
        # print(result_py)
        np.testing.assert_allclose(result, result_py)
        print('all results close')

        */

        // parameters for a standard normal (mean=0, sigma=1)
        let mu = na::Vector2::<f64>::new(0.0, 0.0);
        let precision = na::Matrix2::<f64>::new(1.0, 0.0, 0.0, 1.0);

        let mvn = MultivariateNormal::from_mean_and_precision(&mu, &precision);

        let xs = na::OMatrix::<f64, na::U2, na::U3>::new(0.0, 1.0, 0.0, 0.0, 0.0, 1.0).transpose();

        let results = mvn.pdf(&xs);

        // check single equals vectorized form
        for i in 0..xs.nrows() {
            let x = xs.row(i).clone_owned();
            let di = mvn.pdf(&x)[0];
            assert_relative_eq!(di, results[i], epsilon = 1e-10);
        }

        dbg!((results[0], 0.15915494));

        let epsilon = 1e-5;
        // some spot checks with standard normal
        assert_relative_eq!(results[0], 0.15915494, epsilon = epsilon);
        assert_relative_eq!(results[1], 0.09653235, epsilon = epsilon);
        assert_relative_eq!(results[2], 0.09653235, epsilon = epsilon);
    }
}
