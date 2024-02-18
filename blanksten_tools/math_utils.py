import numpy as np


def parse_matrix(string, dim=None, sep=" "):
    flat = np.array([float(i) for i in string.split(sep) if i.replace(".","1").replace("-","1").isnumeric()])
    if dim is None:
        return flat
    if "sq" in dim:
        from math import sqrt
        dim = int(sqrt(len(flat))), int(sqrt(len(flat)))
    M = flat.reshape(dim)
    return M

def linearLSQ(A,y):
    Q,R=np.linalg.qr(A,mode="reduced")
    x=np.linalg.lstsq(R,Q.T@y)
    return x

def rot_2d(theta, degrees = False):
    if degrees:
        theta = np.radians(theta)
    sint = np.sin(theta)
    cost = np.cos(theta)
    R = np.array([[cost, -sint], [sint, cost]])
    return R

def transform_point(p, s, R, t):
    if not hasattr(R, "__len__"):
        R = rot_2d(R)
    q = s * R @ p + t
    return q

def transform_points(points, s, R, t):
    q = np.array([transform_point(pi, s, R, t) for pi in points])
    return q
    
def dist_to_centroid(p, avg = False, return_centroid = False):
    mu = np.mean(p, axis=0)
    dist = np.sqrt(np.sum((p - mu)**2,axis=1))
    if avg:
        dist = np.mean(dist)
    if return_centroid: return dist, mu
    return dist

def centroid(p):
    mu = np.mean(p, axis=0)
    return mu

def euclidean_norm(a):
    return np.sqrt(np.dot(a, a))

def avg_dist_to_point(points, p):
    return np.mean([euclidean_norm(points[i] - p) for i in range(n)])

def covariance_matrix(p, q, mup = False, muq = None):
    if mup is None:
        mup = centroid(p)
    if muq is None:
        muq = centroid(q)
    C = np.sum([np.outer((q - muq)[i], (p - mup)[i]) for i in range(n)], axis=0)
    return C

def find_transform(p, q):
    """
    Finds scale (s), rotation (R) and translation (t), such that:
    q - (s * R * p + t)
    is minimized.
    :param p: (n, 2)-np.array. Points pre-transformation.
    :param q: (n, 2)-np.array. Points post-transformation.
    :return s: Scale
    :return R: Rotation matrix.
    :return t: Translation.
    """
    if not np.array_equal(p.shape, q.shape):
        return ":("
    if p.shape[1] != 2:
        if p.shape[0] == 2:
            p = p.T
            q = q.T
        else:
            return ":("

    n = p.shape[1] 
    mup = centroid(p)
    muq = centroid(q)
    s = avg_dist_to_point(q, muq) / avg_dist_to_point(p, mup)

    C = covariance_matrix(p, q, mup, muq)

    U, S, Vt = np.linalg.svd(C)
    Rh = U @ Vt
    D = np.array([[1, 0], [0, np.linalg.det(Rh)]])

    R = Rh @ D
    t = muq - s * R @ mup

    return s, R, t


def gauss(x, sigma):
    return 1/np.sqrt(2 * np.pi * sigma**2) * np.exp(-x**2/(2 * sigma**2))

def gauss_deriv(x, sigma):
    return -x/(sigma**3 * np.sqrt(2* np.pi)) * np.exp(-x**2/(2*sigma**2))

