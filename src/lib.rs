//! Multivariate normal distribution using [nalgebra](https://nalgebra.org).
//!
//! # Example of usage
//!
//! ```
//! use nalgebra::{ Vector2, Matrix2, MatrixMN, U2, U3};
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
//! let xs = MatrixMN::<_,U3,U2>::new(
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

use nalgebra::{VectorN, MatrixMN, DefaultAllocator, Dim, RealField, linalg,
    allocator::Allocator};

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
#[derive(Debug,Clone)]
pub enum ErrorKind {
    NotDefinitePositive,
}

/// An `N`-dimensional multivariate normal distribution
///
/// See the [crate-level docs](index.html) for example usage.
#[derive(Debug,Clone)]
pub struct MultivariateNormal<Real,N>
    where
        Real: RealField,
        N: Dim + nalgebra::DimMin<N, Output = N>,
        DefaultAllocator: Allocator<Real, N>,
        DefaultAllocator: Allocator<Real, N, N>,
        DefaultAllocator: Allocator<Real, nalgebra::U1, N>,
        DefaultAllocator: Allocator<(usize, usize), <N as nalgebra::DimMin<N>>::Output>,
{
    /// Negative of mean of the distribution
    neg_mu: nalgebra::VectorN<Real,N>,
    /// Precision of the distribution (the inverse of covariance)
    precision: nalgebra::MatrixN<Real,N>,
    /// A cached value used for calculating the density
    fac: Real,
}

impl<Real,N> MultivariateNormal<Real,N>
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
        mu: &nalgebra::VectorN<Real,N>,
        precision: &nalgebra::MatrixN<Real,N>,
    ) -> Self {
        // Here we calculate and cache `fac` to prevent computing it repeatedly.

        // The determinant of the inverse of an invertible matrix is
        // the inverse of the determinant.
        let precision_det = nalgebra::linalg::LU::new(precision.clone()).determinant();
        let det = Real::one()/precision_det;

        let ndim = mu.nrows();
        let fac: Real = Real::one() / ( Real::two_pi().powi(ndim as i32)
            * det.abs() ).sqrt();

        Self { neg_mu: -mu, precision: precision.clone(), fac }
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
        mu: &nalgebra::VectorN<Real,N>,
        covariance: &nalgebra::MatrixN<Real,N>,
    ) -> Result<Self,Error> {
        // calculate precision from covariance.
        let precision = linalg::Cholesky::new(covariance.clone())
            .ok_or(Error{kind: ErrorKind::NotDefinitePositive})?.inverse();
        let result = Self::from_mean_and_precision(mu, &precision);
        Ok(result)
    }

    fn inner_pdf<Count>(
        &self,
        xs_t: &nalgebra::MatrixMN<Real,Count,N>,
    ) -> nalgebra::VectorN<Real,Count>
        where
            Count: Dim,
            DefaultAllocator: Allocator<Real, Count>,
            DefaultAllocator: Allocator<Real, N, Count>,
            DefaultAllocator: Allocator<Real, Count, N>,
            DefaultAllocator: Allocator<Real, Count, Count>,
    {
        let dvs: nalgebra::MatrixMN<Real,Count,N> = broadcast_add(&xs_t,&self.neg_mu);

        let left: nalgebra::MatrixMN<Real,Count,N> = &dvs * &self.precision;
        let ny2_tmp: nalgebra::MatrixMN<Real,Count,N> = left.component_mul( &dvs );
        let ones = nalgebra::MatrixMN::<Real,N,nalgebra::U1>::repeat_generic(
            N::from_usize(self.neg_mu.nrows()),
            nalgebra::U1,
            nalgebra::convert::<f64,Real>(1.0),
            );
        let ny2: nalgebra::VectorN<Real,Count> = ny2_tmp * ones;
        let y: nalgebra::VectorN<Real,Count> = ny2 * nalgebra::convert::<f64,Real>(-0.5);
        y

    }

    /// Probability density function
    ///
    /// Evaluate the probability density at locations `xs`.
    pub fn pdf<Count>(
        &self,
        xs: &nalgebra::MatrixMN<Real,Count,N>,
    ) -> nalgebra::VectorN<Real,Count>
        where
            Count: Dim,
            DefaultAllocator: Allocator<Real, Count>,
            DefaultAllocator: Allocator<Real, N, Count>,
            DefaultAllocator: Allocator<Real, Count, N>,
            DefaultAllocator: Allocator<Real, Count, Count>,
    {
        let y = self.inner_pdf(xs);
        vec_exp(&y) * self.fac
    }

    /// Log of the probability density function
    ///
    /// Evaluate the log probability density at locations `xs`.
    pub fn logpdf<Count>(
        &self,
        xs: &nalgebra::MatrixMN<Real,Count,N>,
    ) -> nalgebra::VectorN<Real,Count>
        where
            Count: Dim,
            DefaultAllocator: Allocator<Real, Count>,
            DefaultAllocator: Allocator<Real, N, Count>,
            DefaultAllocator: Allocator<Real, Count, N>,
            DefaultAllocator: Allocator<Real, Count, Count>,
    {
        let y = self.inner_pdf(xs);
        vec_add(&y,self.fac.ln())
    }
}

fn vec_exp<Real,Count>(v: &nalgebra::VectorN<Real,Count>) -> nalgebra::VectorN<Real,Count>
    where
        Real: RealField,
        Count: Dim,
        DefaultAllocator: Allocator<Real, Count>,
{
    let nrows = Count::from_usize(v.nrows());
    VectorN::from_iterator_generic(nrows, nalgebra::U1,
        v.iter().map(|vi| vi.exp()))
}

fn vec_add<Real,Count>(v: &nalgebra::VectorN<Real,Count>, rhs: Real) -> nalgebra::VectorN<Real,Count>
    where
        Real: RealField,
        Count: Dim,
        DefaultAllocator: Allocator<Real, Count>,
{
    let nrows = Count::from_usize(v.nrows());
    VectorN::from_iterator_generic(nrows, nalgebra::U1,
        v.iter().map(|vi| *vi + rhs))
}

/// Add `vec` to each row of `arr`, returning the result with shape of `arr`.
///
/// Inputs `arr` has shape R x C and `vec` is C dimensional. Result
/// has shape R x C.
fn broadcast_add<Real, R, C>(arr: &MatrixMN<Real,R,C>, vec: &VectorN<Real,C>) -> MatrixMN<Real,R,C>
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
    MatrixMN::from_iterator_generic( nrows, ncols,
        arr.iter().enumerate().map(|(i,el)| {
            let vi = i/ndim; // integer div to get index into vec
            *el+vec[vi]
        } )
    )
}

#[cfg(test)]
mod tests {
    use nalgebra as na;
    use approx::relative_eq;
    use crate::*;

    /// Calculate the sample covariance
    ///
    /// Calculates the sample covariances among K variables based on N observations
    /// each. Calculates K x K covariance matrix from observations in `arr`, which
    /// is N rows of K columns used to store N vectors of dimension K.
    fn sample_covariance<Real: RealField, N: Dim, K: Dim>(arr: &MatrixMN<Real,N,K>) -> nalgebra::MatrixN<Real,K>
        where DefaultAllocator: Allocator<Real, N, K>,
            DefaultAllocator: Allocator<Real, K, N>,
            DefaultAllocator: Allocator<Real, K, K>,
            DefaultAllocator: Allocator<Real, K>,
    {
        let mu: VectorN<Real,K> = mean_axis0(arr);
        let y = broadcast_add(arr,&-mu);
        let n: Real = Real::from_usize(arr.nrows()).unwrap();
        let sigma = (y.transpose() * y) / (n - Real::one());
        sigma
    }

    /// Calculate the mean of R x C matrix along the rows and return C dim vector
    fn mean_axis0<Real, R, C>(arr: &MatrixMN<Real,R,C>) -> VectorN<Real,C>
        where
            Real: RealField,
            R: Dim,
            C: Dim,
            DefaultAllocator: Allocator<Real, R, C>,
            DefaultAllocator: Allocator<Real, C>,
    {
        let vec_dim: C = C::from_usize(arr.ncols());
        let mut mu = VectorN::<Real,C>::zeros_generic(vec_dim, nalgebra::U1);
        let scale: Real = Real::one()/na::convert(arr.nrows() as f64);
        for j in 0..arr.ncols() {
            let col_sum = arr
                .column(j)
                .iter()
                .fold(Real::zero(), |acc, &x| acc + x);
            mu[j] = col_sum * scale;
        }
        mu
    }

    #[test]
    fn test_covar() {
        use nalgebra::core::dimension::{U2, U3};

        let arr = MatrixMN::<f64,U3,U2>::new(
            1.0, 0.1,
            2.0, 0.2,
            3.0, 0.3,
        );

        let c = sample_covariance(&arr);

        let expected = nalgebra::MatrixN::<f64,U2>::new(
            0.2, 0.1,
            0.1, 0.01,
        );

        relative_eq!(c, expected);
    }

    #[test]
    fn test_mean_axis0() {
        use nalgebra::core::dimension::{U2, U4};

        let a1 = MatrixMN::<f64,U2,U4>::new(1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0);
        let actual1: VectorN<f64,U4> = mean_axis0(&a1);
        let expected1 = &[ 3.0, 4.0, 5.0, 6.0 ];
        assert!(actual1.as_slice() == expected1);

        let a2 = MatrixMN::<f64,U4,U2>::new(1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0);
        let actual2: VectorN<f64,U2> = mean_axis0(&a2);
        let expected2 = &[ 4.0, 5.0 ];
        assert!(actual2.as_slice() == expected2);
    }

    #[test]
    fn test_broadcast_add() {
        use nalgebra::core::dimension::{U3, U4};

        let x = MatrixMN::<f64,U3,U4>::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            100.0, 200.0, 300.0, 400.0,
            );
        let v = VectorN::<f64,U4>::new(-3.0, -4.0, -5.0, -3.0);
        let actual = broadcast_add(&x, &v);

        let expected = MatrixMN::<f64,U3,U4>::new(
            -2.0, -2.0, -2.0, 1.0,
            2.0, 2.0, 2.0, 5.0,
            97.0, 196.0, 295.0, 397.0,
            );

        assert!(actual == expected);
    }

    #[test]
    fn test_density() {
        // parameters for a standard normal (mean=0, sigma=1)
        let mu = na::Vector2::<f64>::new(0.0,0.0);
        let precision = na::Matrix2::<f64>::new(1.0, 0.0, 0.0, 1.0);

        let mvn = MultivariateNormal::from_mean_and_precision(&mu, &precision);

        let xs = na::MatrixMN::<f64,na::U2,na::U3>::new(
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            ).transpose();

        let results = mvn.pdf(&xs);

        // check single equals vectorized form
        for i in 0..xs.nrows() {
            let x = xs.row(i).clone_owned();
            let di = mvn.pdf(&x)[0];
            relative_eq!( di, results[i], epsilon = 1e-10 );
        }

        // some spot checks with standard normal
        relative_eq!( results[0], 1.0/(2.0*std::f64::consts::PI).sqrt(), epsilon = 1e-10 );
        relative_eq!( results[1], 1.0/(2.0*std::f64::consts::PI).sqrt() * (-0.5f64*1.0f64).exp(), epsilon = 1e-10 );
        relative_eq!( results[2], 1.0/(2.0*std::f64::consts::PI).sqrt() * (-0.5f64*1.0f64).exp(), epsilon = 1e-10 );
    }
}