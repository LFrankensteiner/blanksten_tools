import numpy as np

def insertion_sort(A, n = None):
    if n is None:
        n = len(A)
    for i in range(1, n):
        key = A[i]
        j = i - 1
        while (j >= 0) and (A[j] > key):
            A[j+1] = A[j]
            j = j - 1
        A[j+1] = key
    return A

def insertion_sort_inv(A, n = None):
    if n is None:
        n = len(A)
    for i in range(1, n):
        key = A[i]
        j = i - 1
        while (j >= 0) and (A[j] < key):
            A[j+1] = A[j]
            j = j - 1
        A[j+1] = key
    return A

def merge(A, p, q, r):
    nl = q - p
    nr = r - q
    L = copy(A[p:q])
    R = copy(A[q:r])
    i = 0
    j = 0
    k = p
    while (i < nl) and (j < nr):
        if L[i] <= R[j]:
            A[k] = L[i]
            i = i + 1
        else:
            A[k] = R[j]
            j = j + 1
        k = k + 1
    while i < nl:
        A[k] = L[i]
        i = i + 1
        k = k + 1
    while j < nr:
        A[k] = R[j]
        j = j + 1
        k = k + 1
        
    return A


def LSF(x, y, functup):
    At = []
    for f in functup:
        At.append(np.array(f(x)).flatten())
    At = np.array(At)
    A = At.T

    c = np.linalg.solve(At.dot(A), At.dot(y))
    xx = np.linspace(x[0], x[-1], 100)
    yy = np.zeros(len(x))
    for i,f in enumerate(functup):
        yy += c[i] * f(xx)

    plt.plot(x, y, 'o', xx, yy, '-')
    plt.title(f'{np.round(c.T[0],6)}')

    plt.show()

def bisection(f, a, b, nmax, fejl = False):
    # Check input.
    if nmax < 1:
        raise ValueError("nmax must be a positive number")

    # Prepare to iterate.
    if a >= b:
        raise ValueError("a must be less than b")
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    X = [0] * (nmax + 1)  # Create X to store the iterations.

    # Iterate.
    for n in range(nmax):
        c = (a + b) / 2  # The midpoint.
        fc = f(c)  # The corresponding function value.
        X[n] = c
        if fa * fc < 0:
            b = c
            # fb = fc  # Note that fb = f(b) is actually not used.
        else:
            a = c
            fa = fc

    # Finish by computing the midpoint of the last interval.
    c = (a + b) / 2
    X[nmax] = c
    if not fejl:
        return X
    E = abs((b - a)/2**(n+1))
    return [X, E]
    
def newton(f, df,x0,nmax):
    X = []
    x = x0
    X.append(x)
    for i in range(nmax):
        fx = f(x)
        fp = df(x)
        x = x - fx/fp
        X.append(x)
    return X

def secant(f, x0, x1, nmax):
    # Set a tolerance for the error
    tol = 1e-6
    X = [x1]
    # Iterate nmax times
    for n in range(nmax):
        # Compute the value of the function at x0 and x1
        f0 = f(x0)
        f1 = f(x1)
        
        # Compute the secant of the function at x0 and x1
        sec = (x1 - x0) / (f1 - f0)
        
        # Compute the next guess for the root
        x2 = x1 - sec * f1
        X.append(x2)
        # Check if the error is within the tolerance
        if abs(x2 - x1) < tol:
            return X
        
        # Update x0 and x1 for the next iteration
        x0 = x1
        x1 = x2
    
    # If no root was found, return None
    return X

def euler(dxdt, tspan, x0, n):
    a, b = tspan
    t = np.linspace(a, b, n+1) # Dette er bedre end at fremskrive t
    h = (b - a) / n # h is calculated once only
    x = np.zeros(n+1) # preallokere x det er hurtigere
    x[0] = x0 # lægger x0 som første x-værdi
    for i in range(n): # implementerer Euler's method
        x[i+1] = x[i] + dxdt(t[i], x[i]) * h
    return t, x

def taylor(f, x0, n, fejl = False):
    """
    Modtager sympy expression "f" som skal være en funktion af x. x0 som skal være udviklingspunktet, og graden.
    Returnerer taylorpol.
    Hvis fejl = True, returneres liste bestående af pol. og rest.
    """
    x = symbols("x")
    TaylorPol = 0
    for n in range(n+1):
        TaylorPol += f.diff(x, n).subs(x, x0)/factorial(n) * (x - x0)**n
    if not fejl:
        return TaylorPol
    TaylorRest = (f.diff(x, n+1)/factorial(n + 1) * (x - x0)**(n+1)).subs(x, S("Q"))
    return [TaylorPol, TaylorRest]

def euler_system(dxdt, tspan, x0, n):
    
    dim1 = x0.shape[0]
    dim2 = dxdt(tspan[0], x0).shape[0]  # a bit wasteful, but can be saved
    if dim1 - dim2 != 0:
        raise ValueError('The dimensions of x0 and the right side do not match')

    a, b = tspan
    
    t = np.linspace(a, b, n+1)
    h = (b - a) / n  # h is calculated once only
    x = np.zeros((dim1, n+1))  # preallocate x to improve efficiency
    x[:,0] = x0
    for i in range(n):  # Euler's method
        x[:,i+1] = x[:,i] + dxdt(t[i], x[:,i]) * h
    return t, x
