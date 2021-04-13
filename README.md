# nalgebra-mvn

[![Crates.io](https://img.shields.io/crates/v/nalgebra-mvn.svg)](https://crates.io/crates/nalgebra-mvn)

Multivariate normal distribution using [nalgebra](https://nalgebra.org).

## Example of usage

```rust
use nalgebra::{ Vector2, Matrix2, OMatrix, U2, U3};
use nalgebra_mvn::MultivariateNormal;

// specify mean and covariance of our multi-variate normal
let mu = Vector2::from_row_slice(&[9.0, 1.0]);
let sigma = Matrix2::from_row_slice(
    &[1.0, 0.0,
    0.0, 1.0]);

let mvn = MultivariateNormal::from_mean_and_covariance(&mu, &sigma).unwrap();

// input samples are row vectors vertically stacked
let xs = OMatrix::<_,U3,U2>::new(
    8.9, 1.0,
    9.0, 1.0,
    9.1, 1.0,
);

// evaluate the density at each of our samples.
let result = mvn.pdf(&xs);

// result is a vector with num samples rows
assert!(result.nrows()==xs.nrows());
```

## License
Licensed under either of

* Apache License, Version 2.0,
  (./LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license (./LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.

License: MIT/Apache-2.0
