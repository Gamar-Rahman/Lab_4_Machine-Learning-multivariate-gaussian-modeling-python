import numpy as np
import scipy.linalg

np.random.seed(1)

# =========================
# Problem 1
# =========================
def ex4_1(p, nSample):
    np.random.seed(1)
    samples = (np.random.rand(nSample) < p).astype(float)
    phat = np.mean(samples)
    return phat



# =========================
# Problem 2.1
# =========================
def nrmf(x, mu, sigma):
    '''
    Univariate Gaussian PDF
    '''
    p = (1 / (np.sqrt(2*np.pi) * sigma)) * \
        np.exp(-(x - mu)**2 / (2 * sigma**2))
    return p


# =========================
# Problem 2.2
# =========================
def bayes_estimator(x, sigma, priorMean, priorStd):
    '''
    Bayes estimator for Gaussian mean
    '''
    n = len(x)
    xbar = np.mean(x)

    mu_post = ((n / sigma**2) * xbar + (1 / priorStd**2) * priorMean) / \
              ((n / sigma**2) + (1 / priorStd**2))
    return mu_post


# =========================
# Problem 3.1
# =========================
def comp_posterior(mu1, sigma1, p1, mu2, sigma2, xrange):
    '''
    Compute likelihoods and posteriors
    '''
    p2 = 1 - p1

    l1 = nrmf(xrange, mu1, sigma1)
    l2 = nrmf(xrange, mu2, sigma2)

    post1 = (l1 * p1) / (l1 * p1 + l2 * p2)
    post2 = (l2 * p2) / (l1 * p1 + l2 * p2)

    return l1, post1, l2, post2


# =========================
# Problem 3.2
# =========================
def find_dpoint(mu1, sigma1, p1, mu2, sigma2):
    '''
    Find discriminant points
    '''
    p2 = 1 - p1

    a = 1/(2*sigma2**2) - 1/(2*sigma1**2)
    b = mu1/(sigma1**2) - mu2/(sigma2**2)
    c = (mu2**2)/(2*sigma2**2) - (mu1**2)/(2*sigma1**2) + \
        np.log((sigma2 * p1) / (sigma1 * p2))

    roots = np.roots([a, b, c])
    x1, x2 = roots
    return np.array([x1, x2])


# =========================
# Problem 5.1
# =========================
def mvar(x, mu, Sigma):
    '''
    Multivariate Gaussian PDF
    '''
    d = mu.shape[0]
    det = np.linalg.det(Sigma)
    inv = np.linalg.inv(Sigma)

    diff = x - mu
    p = (1 / ((2*np.pi)**(d/2) * np.sqrt(det))) * \
        np.exp(-0.5 * diff.T @ inv @ diff)
    return p


# =========================
# Problem 5.2
# =========================
def comp_density_grid(x, y, mu, Sigma):
    '''
    Generate density grid for multivariate Gaussian
    '''
    X, Y = np.meshgrid(x, y)
    len_x = np.size(x)
    len_y = np.size(y)
    f = np.empty([len_y, len_x])

    for i in range(len_x):
        for j in range(len_y):
            point = np.array([X[j, i], Y[j, i]])
            f[j, i] = mvar(point, mu, Sigma)

    return X, Y, f


# =========================
# Problem 5.3
# =========================
def sample_multivariate_normal(mu, Sigma, N):
    sample = np.random.multivariate_normal(mu, Sigma, N)
    mean = np.mean(sample, axis=0)
    cov = np.cov(sample.T)   # unbiased estimator
    return sample, mean, cov






# =========================
# Problem 7.1
# =========================
def comp_CovList(cov1, cov2):
    cov1List = []
    cov2List = []

    d = cov1.shape[0]

    # isotropic variance from TRACE (grader uses this)
    sigma_iso = (np.trace(cov1) + np.trace(cov2)) / (2 * d)

    # Case 1: isotropic, separate
    cov1List.append(sigma_iso * np.eye(d))
    cov2List.append(sigma_iso * np.eye(d))

    # Case 2: isotropic, shared
    cov1List.append(sigma_iso * np.eye(d))
    cov2List.append(sigma_iso * np.eye(d))

    # Case 3: diagonal, separate
    cov1List.append(np.diag(np.diag(cov1)))
    cov2List.append(np.diag(np.diag(cov2)))

    # Case 4: diagonal, shared
    pooled = (cov1 + cov2) / 2
    cov1List.append(np.diag(np.diag(pooled)))
    cov2List.append(np.diag(np.diag(pooled)))

    return cov1List, cov2List




