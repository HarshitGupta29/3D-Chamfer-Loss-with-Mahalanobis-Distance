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

## Implementation Details
- Input to the function: $\( \mu, Y, L \)$
  - $\( \mu \)$ is a 1xNx3 tensor representing the mean vectors of $\( N \)$ Gaussians.
  - $\( Y \)$ is a 1xMx3 tensor representing $\( M \)$ points in a point cloud.
  - $\( L \)$ is a 1xNx6 tensor representing the lower triangular matrix of the Cholesky decomposition of each Gaussian's covariance matrix, flattened into a 1x6 vector.
- The function outputs all the distances between points in $\( Y \)$ and the Gaussians in $\( X \)$.
- The final Chamfer Loss is computed by applying `torch.mean` to the sum of the minimum distances for both $\( X \)$ and $\( Y \)$.

## Installation
Installation instructions for the Chamfer Loss with Mahalanobis Distance can be found at the original repository: [ThibaultGROUEIX/ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch).

## Contributing to the Project
We welcome contributions and improvements to the Chamfer Loss implementation. If you have suggestions for enhancements or have identified issues, please consider the following guidelines for contributing:

- **Pull Requests**: If you wish to contribute code, ensure that your PR is accompanied by a clear description of the problem and a detailed explanation of the proposed solution. Code contributions should be thoroughly tested before submission.

- **Suggestions and Improvements**: We appreciate constructive feedback and innovative ideas. If you have a suggestion, please provide a clear and detailed explanation, ideally with a proposed solution or actionable plan. This helps in understanding the context and potential impact of your suggestion.

- **Feedback and Issues**: While feedback is valuable, we encourage contributors to offer solutions or actionable suggestions alongside their feedback.

Remember, respectful and constructive communication fosters a productive and welcoming environment for all contributors.
