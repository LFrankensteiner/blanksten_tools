from .imgproc import apply_gauss_deriv
import numpy as np
import matplotlib as plt
import cv2 as cv
from skimage import color

def linearLSQ(A,y):
    Q,R=np.linalg.qr(A,mode="reduced")
    x=np.linalg.solve(R,Q.T@y)
    return x

def read_video_cv(video_path):
    cap = cv.VideoCapture(video_path)
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    i = 0
    dims = cap.read()[1].shape
    allFrames = np.zeros([*dims[:2], n_frames])
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:, :, ::-1]
        cf = color.rgb2gray(frame)
        allFrames[:,:,i] = cf
        i += 1
    cap.release()
    return allFrames


class OpticFlowVid:
    def __init__(self, V, sigma, s):
        self.V = V
        self.sigma = sigma
        self.s = s
        self.x_dim, self.y_dim, self.frames = V.shape
        self.gdx = apply_gauss_deriv(V, sigma=sigma, s=s, axis=1)
        self.gdy = apply_gauss_deriv(V, sigma=sigma, s=s, axis=0)
        self.gdt = apply_gauss_deriv(V, sigma=sigma, s=s, axis=2)

    
    def frame(self, frame):
        return self.V[:,:,frame]

    def optic_flow_voxel(self,N, xi, yi, frame):
        Vxi = self.gdx[xi-N:xi+N+1, yi-N:yi+N+1, frame].flatten()
        Vyi = self.gdy[xi-N:xi+N+1, yi-N:yi+N+1, frame].flatten()
        Vti = self.gdt[xi-N:xi+N+1, yi-N:yi+N+1, frame].flatten()
        A = np.array([Vxi, Vyi]).T
        xy = linearLSQ(A, -Vti)
        return xy
    
    def optic_flow_grid(self, N, xstride, ystride):
        flow = dict.fromkeys(range(self.frames))
        for i in range(self.frames):
            flow[i] = []
            for xi in range(max(N,xstride//2), self.x_dim, xstride):
                for yi in range(max(N,ystride//2), self.y_dim, ystride):
                    dxdy = self.optic_flow_voxel(N, xi, yi, i)
                    flow[i].append({"pixel" : np.array([xi,yi]), "displacement" : dxdy})
        return flow

    def optic_flow_vid(self, N, xstride, ystride, min_len):
        flow = self.optic_flow_grid(N, xstride, ystride)
        for frame in range(self.frames):
            plt.cla()
            plt.imshow(self.frame(frame))
            flowi = flow[frame]
            for arr in flowi:
                if np.dot(*arr["displacement"]*2) > min_len:
                    x, y = arr["pixel"]
                    plt.arrow(*arr["pixel"], *arr["displacement"], length_includes_head=True, head_width=3, head_length=1)
            plt.pause(1/1000) 

    def optic_flow_from_grid(self, flow, min_len, interval = 1/1000):
        for frame in range(self.frames):
            plt.cla()
            plt.xlim(0,self.x_dim)
            plt.ylim(self.y_dim,0)
            io.imshow(self.frame(frame))
            flowi = flow[frame]
            for arr in flowi:
                if np.dot(*arr["displacement"]*2) > min_len:
                    x, y = arr["pixel"]
                    plt.arrow(y, x, *arr["displacement"].T, length_includes_head=True, head_width=3, head_length=1, color="cyan")
            plt.pause(interval) 