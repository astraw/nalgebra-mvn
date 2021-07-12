use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra::{Matrix2, OMatrix, Vector2, U2, U3};
use nalgebra_mvn::MultivariateNormal;

fn bench_pdf(c: &mut Criterion) {
    // specify mean and covariance of our multi-variate normal
    let mu = Vector2::from_row_slice(&[9.0, 1.0]);

    {
        let precision = Matrix2::from_row_slice(&[1.0, 0.0, 0.0, 1.0]);

        c.bench_function("create_from_precision", move |b| {
            b.iter(|| {
                MultivariateNormal::from_mean_and_precision(&mu, &precision);
            })
        });
    }

    let covariance = Matrix2::from_row_slice(&[1.0, 0.0, 0.0, 1.0]);

    c.bench_function("create_from_covariance", move |b| {
        b.iter(|| {
            MultivariateNormal::from_mean_and_covariance(&mu, &covariance).unwrap();
        })
    });

    let mvn = MultivariateNormal::from_mean_and_covariance(&mu, &covariance).unwrap();

    {
        let mvn = mvn.clone();
        let xs = OMatrix::<f64, nalgebra::Dynamic, U2>::from_fn(
            100000,
            |row, col| {
                if col == 0 {
                    row as f64
                } else {
                    1.0
                }
            },
        );

        c.bench_function("pdf_big", move |b| b.iter(|| mvn.pdf(&xs)));
    }

    {
        let mvn = mvn.clone();
        // input samples are row vectors vertically stacked
        let xs = OMatrix::<_, U3, U2>::new(8.9, 1.0, 9.0, 1.0, 9.1, 1.0);

        c.bench_function("pdf_3x2", move |b| b.iter(|| mvn.pdf(&xs)));
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_pdf,
}
criterion_main!(benches);
