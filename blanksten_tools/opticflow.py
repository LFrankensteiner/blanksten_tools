import numpy as np
from scipy.ndimage import convolve1d
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import color, io
import os
from .math_utils import linearLSQ
from .math_utils import gauss, gauss_deriv
from .imgproc import dist_arr, gauss_kernel1d, gauss_deriv_kernel1d, apply_gauss, apply_gauss_deriv

def read_img_folder(folder, gray = True):
    frame_names = os.listdir(folder) # List of files in folder of images
    n = len(frame_names) # Number of frames
    # Preallocates array to store frames
    frame = frame_names[0]
    im = io.imread(os.path.join(folder,frame))
    if gray:
        V = np.zeros([*im.shape[:2],n])
        for i,frame in enumerate(frame_names):
            im = io.imread(os.path.join(folder,frame))
            g = color.rgb2gray(im)
            V[:,:,i] = g
    else:
        V = np.zeros([*im.shape[:2],3,n], int)
        for i,frame in enumerate(frame_names):
            im = io.imread(os.path.join(folder,frame))
            V[:,:,:,i] = im
    return V


def read_video_cv(video_path, n_frames = None):
    cap = cv.VideoCapture(video_path)
    if n_frames is None:
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

def play_vid(vid, frame_axis = 2):
    if frame_axis == 0:
        vid = np.transpose(vid, (2, 0, 1))
    if frame_axis == 1:
        vid = np.transpose(vid, (0, 2, 1))
    n = vid.shape[2]
    for i in range(n):
        cv.imshow("img",vid[:,:,i])
        if cv.waitKey(10) == ord('q'):
            break
    cv.destroyAllWindows()

# Class to contain a optic flow object.
class OpticFlowVid:
    def __init__(self, V, sigma, s):
        self.V = V
        self.sigma = sigma
        self.s = s
        self.x_dim, self.y_dim, self.frames = V.shape
        self.gdx = apply_gauss_deriv(V, sigma=sigma, s=s, axis=1)
        self.gdy = apply_gauss_deriv(V, sigma=sigma, s=s, axis=0)
        self.gdt = apply_gauss_deriv(V, sigma=sigma, s=s, axis=2)
        self.vid_grid = np.concatenate([np.concatenate([self.V, self.gdx], axis=1), np.concatenate([self.gdy, self.gdt], axis=1)])


    
    def frame(self, framei):
        return self.V[:,:,framei]

    def optic_flow_voxel(self,N, xi, yi, frame):
        Vxi = self.gdx[xi-N:xi+N+1, yi-N:yi+N+1, frame].flatten()
        Vyi = self.gdy[xi-N:xi+N+1, yi-N:yi+N+1, frame].flatten()
        Vti = self.gdt[xi-N:xi+N+1, yi-N:yi+N+1, frame].flatten()
        A = np.array([Vxi, Vyi]).T
        xy, rank = linearLSQ(A, -Vti)
        return xy, rank
    

    def optic_flow_frame(self, N, xstride, ystride, frame): 
        flow = []
        for xi in range(max(N,xstride//2), self.x_dim-N, xstride):
            for yi in range(max(N,ystride//2), self.y_dim-N, ystride):
                dxdy, rank = self.optic_flow_voxel(N, xi, yi, frame)
                if rank == 2:
                    flow.append({"pixel" : np.array([xi,yi]), "displacement" : dxdy})
        return flow

    def optic_flow_plot_frame(self, framei, flow, min_len = 0):
        """
        Plays optic flow video from precomputed flow grid.

        """

        plt.cla()
        plt.ylim(self.x_dim,0)
        plt.xlim(0,self.y_dim)
        io.imshow(self.frame(framei))
        for arr in flow:
            if np.dot(*arr["displacement"]*2) > min_len:
                x, y = arr["pixel"]
                plt.arrow(y, x, *arr["displacement"].T, length_includes_head=True, head_width=5, head_length=2, color="cyan")

                
    def optic_flow_grid(self, N, xstride, ystride):
        flow = dict.fromkeys(range(self.frames))
        for i in range(self.frames):
            flow[i] = self.optic_flow_frame(N, xstride, ystride, i)
        return flow

    def optic_flow_vid(self, N, xstride, ystride, min_len = 0,  interval = 1/1000):
        flow = self.optic_flow_grid(N, xstride, ystride)
        self.optic_flow_from_grid(flow, min_len, interval)

    def optic_flow_from_grid(self, flow, min_len = 0, interval = 1/1000):
        """
        Plays optic flow video from precomputed flow grid.

        """
        for framei in range(self.frames):
            plt.cla()
            plt.ylim(self.x_dim,0)
            plt.xlim(0,self.y_dim)
            io.imshow(self.frame(framei))
            flowi = flow[framei]
            for arr in flowi:
                if np.dot(*arr["displacement"]*2) > min_len:
                    x, y = arr["pixel"]
                    plt.arrow(y, x, *arr["displacement"].T, length_includes_head=True, head_width=5, head_length=2, color="cyan")
            plt.pause(interval) 