use std::io::Write;

use nalgebra::*;
use nalgebra::allocator::Allocator;

use nalgebra_mvn::MultivariateNormal;

macro_rules! mat_to_vec_string {
    ($name: expr, $arr: expr) => {{
        let mut result: Vec<String> = vec![];

        result.push(format!("{} = np.array([", $name));
        for i in 0..$arr.nrows() {
            let mut line = vec![];
            for j in 0..$arr.ncols() {
                line.push( format!( "{}", $arr[(i,j)] ));
            }
            let l2 = line.join(", ");
            result.push(format!("    [{}],", l2));
        }
        result.push(format!("], dtype=np.float)"));

        result
    }}
}

fn pythonize_expectations<Real,S,Count>(
    xs: &nalgebra::MatrixMN<Real,Count,S>,
    mu: &nalgebra::VectorN<Real,S>,
    covariance: &nalgebra::MatrixN<Real,S>,
    result: &nalgebra::VectorN<Real,Count>,
) -> Vec<String>
    where
        Real: RealField,
        S: Dim + nalgebra::DimMin<S, Output = S>,
        DefaultAllocator: Allocator<Real, S>,
        DefaultAllocator: Allocator<Real, S, S>,
        DefaultAllocator: Allocator<Real, nalgebra::U1, S>,
        DefaultAllocator: Allocator<(usize, usize), <S as nalgebra::DimMin<S>>::Output>,
        Count: Dim,
        DefaultAllocator: Allocator<Real, Count>,
        DefaultAllocator: Allocator<Real, S, Count>,
        DefaultAllocator: Allocator<Real, Count, S>,
        DefaultAllocator: Allocator<Real, Count, Count>,
{
    let xs_lines = mat_to_vec_string!("xs", &xs);
    let mu_lines = mat_to_vec_string!("mu", &mu);
    let covar_lines = mat_to_vec_string!("covariance", &covariance);
    let result_lines = mat_to_vec_string!("result", result);

    let x = vec![xs_lines,
        mu_lines,
        covar_lines,
        result_lines,
        ];
    x.into_iter().flatten().collect()
}

#[test]
fn test_vs_scipy_stats() {
    // specify mean and covariance of our multi-variate normal
    let mu = Vector3::from_row_slice(&[9.0, 1.0, 21.0]);
    let covariance = Matrix3::from_row_slice(
        &[10.5,  4.0, 2.2,
           4.0, 10.4, 3.0,
           2.2,  3.0, 5.0, ]);

    // calculate precision from covariance.
    let precision = linalg::Cholesky::new(covariance)
        .expect("inverting covariance failed")
        .inverse();

    let mvn = MultivariateNormal::from_mean_and_precision(&mu, &precision);

    // input samples are row vectors vertically stacked
    let xs = MatrixMN::<_,U5,U3>::new(
        8.9, 1.0, 21.0,
        9.0, 1.0, 21.0,
        9.1, 1.0, 21.0,
        10.1, 2.2, 21.2,
        -1.1, -2.2, 22.2,
    );

    // evaluate the density at each of our samples.
    let result = mvn.pdf(&xs);

    // --------------------------------------------------------------------
    // write Python file with these results
    // --------------------------------------------------------------------

    let lines = pythonize_expectations(&xs, &mu, &covariance, &result);

    // write expectations to disk

    let start_comments = format!("# This file was auto-generated by {}",file!());
    let lines = lines.join("\n");
    let buf = format!("{}

from scipy.stats import multivariate_normal
import numpy as np

{}

result_py = multivariate_normal.pdf(xs, mean=mu[:,0], cov=covariance)
# print(result_py)
np.testing.assert_allclose(result[:,0], result_py)
print('all results close')
", start_comments, lines);

    let py_fname = "check-mvn.py";

    {
        let mut fd = std::fs::File::create(&py_fname).unwrap();
        fd.write(buf.as_bytes()).unwrap();
        fd.write(b"\n").unwrap();
        fd.sync_all().unwrap();
    }

    // --------------------------------------------------------------------
    // run Python file to check results
    // --------------------------------------------------------------------

    let output = std::process::Command::new("python")
        .arg(py_fname)
        .output()
        .expect("Failed to execute command");

    if !output.status.success() {
        panic!("calling python script at \"{}\" failed", py_fname);
    }

}