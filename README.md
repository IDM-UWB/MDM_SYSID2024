# Measurement Difference Method

The measurement difference method (MDM), like any other correlation method is based on analyzing the residual (or measurement prediction error) vector, which is a linear
function of the state and measurement noise vectors. The linearity of the relation between available residual and the noise vectors, which description is sought, is a key finding that facilitates the identification of noise properties (e.g., the noise covariance matrix) using the least-squares method.

The source codes "MDM_SYSID2024" can be used for the unweighted/weighted/semi-weighted and non/recursive MDM identification techniques.

More details can be found in article [1], [2]

---

> [1] Duník, J., Kost, O., and Straka, O. (2018). Design of measurement difference autocovariance method for estimation of process and measurement noise covariances.
Automatica, 90, 16--24.
> [2] Kost, O., Duník, J., and Straka, O. (2023).
Measurement difference method: A universal tool for noise identification.
IEEE Transactions on Automatic Control, 68(3), 1792--1799.
> [3] Kost, O., Duník, J., and Straka, O. (2024).
Noise Covariances Identification by MDM: Weighting, Recursion, and Implementation.
Submitted to the 20th IFAC Symposium on System Identification (SYSID) 2024.
