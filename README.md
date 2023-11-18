# 3D-Chamfer-Loss-with-Mahalanobis-Distance
**Note:** The initial implementation of the Chamfer Loss is based on the code available at [ThibaultGROUEIX/ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch).

Given two sets $\( X \)$ and $\( Y \)$, where $\( X \)$ is a set of Gaussians, each Gaussian represented as a tuple of mean $\( \mu \)$ and Cholesky whitening matrix $\( L \)$ of the Gaussian, and $\( Y \)$ is a point cloud, the Chamfer Loss is defined as:

$\displaystyle\text{ChamferLoss}(X, Y) = \frac{1}{|X|} \sum_{(\mu, L) \in X} \min_{y \in Y} \lVert L^{-1}(\mu - y) \rVert_2^2 + \frac{1}{|Y|} \sum_{y \in Y} \min_{(\mu, L) \in X} \lVert L^{-1}(\mu - y) \rVert_2^2$


where:
- $\( X \)$ is a set of tuples $\( (\mu,  L) \)$, with $\( \mu \)$ being the mean vector of a Gaussian and $\( L \)$ being the Cholesky decomposition of its covariance matrix.
- $\( Y \)$ is a point cloud.
- $\( |X| \)$ and $\( |Y| \)$ represent the sizes of the sets $\( X \)$ and $\( Y \)$, respectively.
- $\( \lVert L^{-1}(\mu - y) \rVert_2^2 \)$ denotes the squared L2 norm of the transformed vector difference $\( (\mu - y) \)$, transformed by the inverse of the Cholesky matrix $\( L \)$.

**Note:** Gradient with respect to L is not computed at all in this kernel.
