import numpy as np
import sympy as s
from sympy.vector import CoordSys3D
import matplotlib.pyplot as plt

def parse_matrix(string, dim=None, sep=" "):
    flat = np.array([float(i) for i in string.split(sep) if i.replace(".","1").replace("-","1").isnumeric()])
    if dim is None:
        return flat
    if "sq" in dim:
        from math import sqrt
        dim = int(sqrt(len(flat))), int(sqrt(len(flat)))
    M = flat.reshape(dim)
    return M


def present(lhs, expr):
    display(Eq(S(lhs), expr, evaluate=False))
    return


def evaluate(expr,eval):
    if isinstance(eval,tuple):
        expr = expr.subs(*eval)
    else:
        expr = expr.subs(eval)
    expr = s.simplify(expr)
    return expr

class Curve3D:
    def __init__(self,x,y,z, param, param_range):
        self.x = x
        self.y = y
        self.z = z
        self.xyz = s.Matrix([x,y,z])

        self.dp = s.diff(self.xyz, param)
        self.ddp = s.diff(self.dp, param)
        self.dddp = s.diff(self.ddp, param)

        self.param_range = param_range
        self.pmin = param_range[0]
        self.pmax = param_range[1]
        self.param = param

        self.lambdified = s.lambdify(param,self.xyz)

    def jacobi(self, eval=None):
        res = s.simplify(s.sqrt(self.dp.dot(self.dp)))
        res = s.refine(s.refine(abs(res), s.Q.nonnegative(self.pmax - self.param)), s.Q.nonnegative(self.param - self.pmin))
        res = s.trigsimp(res)
        res = s.simplify(res)
        return res

    def v(self, eval=None):
        return s.trigsimp(s.sqrt(self.dp.dot(self.dp)))

    def e(self, eval=None):
        eres = s.simplify(self.dp / self.v())
        return eres
    
    def g(self, eval=None):
        crossp = self.dp.cross(self.ddp)
        denom = s.sqrt(crossp.dot(crossp))
        gres = s.simplify(crossp/denom)
        return gres

    def f(self, eval=None):
        fres = s.simplify(self.g().cross(self.e()))
        return fres

    def frenet_serret(self,eval=None):
        return s.Matrix([self.e(), self.g(), self.f()]).reshape(3,3).T
    
    def curvature(self, eval=None):
        crossp = self.dp.cross(self.ddp)
        k = s.simplify(crossp.dot(crossp)/self.v()**3)
        return k

    def torsion(self,eval=None):
        crossp = self.dp.cross(self.ddp)
        t = s.simplify(crossp.dot(self.dddp)/(crossp.dot(crossp)))
        return t

    def length(self):
        plen = s.integrate(self.jacobi(), (self.param, *self.param_range))
        return s.simplify(plen)

    def plot(self):
        ps = np.linspace(float(self.pmin),float(self.pmax), 200)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        xl = []
        yl = []
        zl = []
        for i in ps:
            x,y,z = self.lambdified(i)
            xl.append(*x)
            yl.append(*y)
            zl.append(*z)
        ax.plot(xl,yl,zl)
        return

    def isregular(self, numeric = False):
        res = s.solve(p.jacobi(), s.Interval(p.pmin, p.pmax), numeric=numeric)
        if len(res) == 0:
            return True
        else:
            return res

class Surface3D:
    def __init__(self,x,y,z, param1, param2, param_range1, param_range2):
        self.x = x
        self.y = y
        self.z = z
        self.xyz = s.Matrix([x,y,z])

        self.ru = s.diff(self.xyz, param1)
        self.rv = s.diff(self.xyz, param2)
        self.ruu = s.diff(self.xyz, param1,param1)
        self.ruv = s.diff(self.xyz,param1, param2)
        self.rvv = s.diff(self.xyz, param2, param2)
        self.param_range1 = param_range1
        self.param_range2 = param_range2
        self.pmin1, self.pmax1 = param_range1
        self.pmin2, self.pmax2 = param_range2
        self.param1 = param1
        self.param2 = param2

        self.lambdified = s.lambdify([param1, param2],self.xyz)

    def jacobi(self, eval=None, assume="positive"):
        crossp = self.ru.cross(self.rv)
        res = s.simplify(s.sqrt(crossp.dot(crossp)))
        if eval is not None:
            res = evaluate(res, eval)
        if assume=="positive":
            return res
        res = s.refine(s.refine(abs(res), s.Q.nonnegative(self.pmax1 - self.param1)), s.Q.nonnegative(self.param1 - self.pmin1))
        res = s.refine(s.refine(res, s.Q.nonnegative(self.pmax2 - self.param2)), s.Q.nonnegative(self.param2 - self.pmin2))
        res = s.trigsimp(res)
        res = s.simplify(res)
        return res
    
    def area(self):
        sarea = s.integrate(self.jacobi(), (self.param1, *self.param_range1), (self.param2, *self.param_range2))
        return s.simplify(sarea)

    def N(self, eval=None):
        crossp = self.ru.cross(self.rv)
        N = crossp / (crossp.dot(crossp))
        if eval is not None:
            N = evaluate(N, eval)
        return s.simplify(N)
    
    def F1(self, eval=None):
        F = s.Matrix([[self.ru.dot(self.ru), self.ru.dot(self.rv)],[self.rv.dot(self.ru), self.rv.dot(self.rv)]])
        if eval is not None:
            F = evaluate(F, eval)
        return s.simplify(F)

    def F2(self,eval=None):
        N = self.N(eval)
        F = s.Matrix([[self.ruu.dot(N), self.ruv.dot(N)],[self.ruv.dot(N), self.rvv.dot(N)]])
        if eval is not None:
            F = evaluate(F, eval)
        F = s.trigsimp(F)
        return s.simplify(F)

    def Weingarten(self,eval=None):
        W = self.F1(eval).inv() @ self.F2(eval)
        if eval is not None:
            W = evaluate(W, eval)
        return s.trigsimp(W)

    def K(self, eval=None):
        K = s.simplify(s.det(self.Weingarten(eval)))
        if eval is not None:
            K = evaluate(K, eval)
        return K

    def H(self, eval=None):
        H = s.simplify(s.S(1)/2 * s.trace(self.Weingarten(eval)))
        H = s.trigsimp(H)
        if eval is not None:
            H = evaluate(H, eval)
        return H

    def plot(self):
        p1s = np.linspace(float(self.pmin1),float(self.pmax1), 200)
        p2s = np.linspace(float(self.pmin2),float(self.pmax2), 200)
        u, v = np.meshgrid(p1s, p2s)
    
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        xl = np.zeros(u.shape)
        yl = np.zeros(u.shape)
        zl = np.zeros(u.shape)
        for i,u in enumerate(p1s):
            for j,v in enumerate(p2s):
                x,y,z = self.lambdified(u,v)
                xl[i,j] = x
                yl[i,j] = y
                zl[i,j] = z
                
        ax.plot_surface(xl,yl,zl, cmap="rainbow")
        return

class Tetrahedron:
    def __init__(self,p, a,b,c):
        self.p = p
        self.a = a
        self.b = b
        self.c = c
        self.M = s.Matrix([a,b,c]).reshape(3,3)

    def Rum(self):
        return s.det(self.M)

    def Vol(self):
        return abs(s.S(1)/6 * self.Rum())
    
    def orientation(self):
        r = self.Rum()
        if r > 0:
            return "positive"
        if r < 0:
            return "negative"
        if r == 0:
            return "non-regular"

    def deform(self, K):
        return K @ self.M # maybe to object?


def RotX(theta):
    M = s.Matrix([[1,0,0], 
    [0, s.cos(theta), -s.sin(theta)],
    [0,s.sin(theta), s.cos(theta)]])
    return M

def RotY(theta):
    M = s.Matrix([[s.cos(theta),0,s.sin(theta)], 
    [0, 1, 0],
    [-s.sin(theta), 0, s.cos(theta)]])
    return M

def RotZ(theta):
    M = s.Matrix([[s.cos(theta),-s.sin(theta),0], 
    [s.sin(theta), s.cos(theta), 0],
    [0,0, 1]])
    return M
