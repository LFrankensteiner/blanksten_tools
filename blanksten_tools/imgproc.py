import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import scipy as sp
from scipy.ndimage import *
from skimage import io, color, segmentation, measure, morphology
from skimage.util import img_as_ubyte
from skimage import img_as_ubyte, img_as_float
import os
from skimage.morphology import erosion, dilation, opening, closing, binary_opening, binary_closing, disk
import math
from skimage.color import label2rgb
from skimage.measure import profile_line
import pydicom as dicom
from skimage.filters import  threshold_otsu, median, prewitt_h, prewitt_v, prewitt, gaussian
from scipy.stats import norm
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from math import ceil
import SimpleITK as sitk
from IPython.display import clear_output

from scipy.linalg import circulant

def gauss(x, sigma):
    return 1/np.sqrt(2 * np.pi * sigma**2) * np.exp(-x**2/(2 * sigma**2))

def gauss_deriv(x, sigma):
    return -x/(sigma**3 * np.sqrt(2* np.pi)) * np.exp(-x**2/(2*sigma**2))

def gauss_2nd_deriv(x, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma**5)*(-sigma**2 + x**2) * np.exp(-x**2/(2*sigma**2))


def dist_arr(s):
    """
    :param s: Parameter determining size of array.
    :return: 1d np.array length 2s+1, like: [-s, -s+1,..., -1, 0, 1,...,s-1, s]
    """
    return np.array([i for i in range(-s, s+1)]) 


def gauss_kernel1d(sigma, s = None):
    if s is None:
        s = 5 * sigma
    s = math.ceil(s)
    x = dist_arr(s)
    return gauss(x, sigma)

def gauss_deriv_kernel1d(sigma, s = None):
    if s is None:
        s = 5 * sigma
    s = math.ceil(s)
    x = dist_arr(s)
    return gauss_deriv(x, sigma)

def gauss_2nd_deriv_kernel1d(sigma, s = None):
    if s is None:
        s = 5 * sigma
    s = math.ceil(s)
    x = dist_arr(s)
    return gauss_2nd_deriv(x, sigma)

def gauss_kernel(sigma, s = None, dim = 2):
    if s is None:
        s = 5 * sigma
    s = math.ceil(s)
    x = dist_arr(s)
    g = gauss(x, sigma)
    ker = g
    for i in range(1, dim):
        ker = np.outer(ker,g)
    ker = ker.reshape(*dim*[2*s+1])
    return ker

def gauss_deriv_kernel(sigma, s = None, dim = 2, axis = 0):
    if s is None:
        s = 5 * sigma
    s = math.ceil(s)
    x = dist_arr(s)
    g = gauss(x, sigma)
    if axis == 0:
        ker = gauss_deriv(x, sigma)
    else:
        ker = g
    for i in range(1,dim):
        if i != axis:
            ker = np.outer(ker, g)
        else:
            ker = np.outer(ker, gauss_deriv(x, sigma))
    return ker.reshape(*[s*2+1]*dim)

def gauss_2nd_deriv_kernel(sigma, s = None, dim = 2, axis = 0):
    if s is None:
        s = 5 * sigma
    s = math.ceil(s)
    x = dist_arr(s)
    g = gauss(x, sigma)
    if axis == 0:
        ker = gauss_2nd_deriv(x, sigma)
    else:
        ker = g
    for i in range(1,dim):
        if i != axis:
            ker = np.outer(ker, g)
        else:
            ker = np.outer(ker, gauss_2nd_deriv(x, sigma))
    return ker.reshape(*[s*2+1]*dim)



"""
def apply_gauss(img, sigma, s = None):
    kernel = gauss_kernel(sigma, s, 2)
    return convolve(img, kernel)

def apply_gauss_deriv(img,sigma, s = None, axis = 0):
    if s is None:
        s = 5 * sigma
    s = math.ceil(s)
    ker = gauss_deriv_kernel(sigma, s, dim = len(img.shape), axis=axis)
    return convolve(img, ker)
"""


def apply_gauss(img, sigma, s = None):
    kernel = gauss_kernel1d(sigma, s)
    for i in range(len(img.shape)):
        img = convolve1d(img, kernel, axis=i)
    return img

def apply_gauss_deriv(img, axis, sigma, s = None):
    kernel = gauss_kernel1d(sigma, s)
    dkernel = gauss_deriv_kernel1d(sigma, s)
    for i in range(len(img.shape)):
        if i == axis:
            img = convolve1d(img, dkernel, axis=i)
        else: 
            img = convolve1d(img, kernel, axis=i)
    return img

def apply_gauss_2nd_deriv(img, axis, sigma, s = None):
    kernel = gauss_kernel1d(sigma, s, 1)
    ddkernel = gauss_2nd_deriv_kernel1d(sigma, s, 1)
    for i in range(len(img.shape)):
        if i == axis:
            img = convolve1d(img, ddkernel, axis=i)
        else: 
            img = convolve1d(img, kernel, axis=i)
    return img

def laplacian(img, sigma, s = None):
    img = img.astype(float)
    ker2 = gauss_2nd_deriv_kernel(sigma, s, dim=1)
    ker1 = gauss_kernel(sigma, s, dim=1)
    #Lxx = convolve1d(convolve1d(img, ker2, axis=1), ker1, axis=0)
    #Lyy = convolve1d(convolve1d(img, ker2, axis=0), ker1, axis=1)
    Lxx = apply_gauss_2nd_deriv(img, 0, sigma, s)
    Lxx = apply_gauss_2nd_deriv(img, 1, sigma, s)
    return Lxx + Lyy


def segmentation_length(img):
    return np.sum(img[1:] != img[:-1]) + np.sum(img[:,1:] != img[:,:-1]) 


def smoothing_kernel(N,a, b=0):
    """
    :param N: Dimension of kernel
    :param a: Weight of first matrix thingy
    :param b: Weight of second matrix thingy. Defaults to 0.
    :return: NxN numpy array of kernel.
    """
    A = circulant([-2,1,*[0]*(N-3),1])
    B = circulant([-6,4,-1,*[0]*(N-5),-1,4])
    return a*A + b*B

def curve_smoothing(X, a, b=0, implicit=False):
    """
    Applies curve smoothing to image.
    :param X: Curve to smooth. Should be a N by 2 np matrix.
    :param a: Weight of first matrix thingy in smoothing kernel lol
    :param b: Weight of second matrix thingy in smoothing kernel lol
    :implicit: Bool. Whether smmothing should be implicit or not.
    """
    if X.shape[0] != 2:
        if X.shape[1] != 2:
            print("??? :(")
            return
        else:
            X = X.T
    N = X.shape[0]
    L = smoothing_kernel(N, a, b)
    if implicit:
        return np.linalg.inv(np.eye(N) - lam * L) @ X
    return (np.eye(N) + L) @ X

def curve_smoothing_plot(X, a, b=0, implicit=False):
    if X.shape[0] != 2:
        if X.shape[1] != 2:
            print("??? :(")
            return
        else:
            X = X.T
    Xnew = curve_smoothing(X, a, b, implicit)
    plt.plot(*X.T)
    plt.plot(*Xnew.T)
    return

def total_variation(img):
    return np.sum(np.abs(img[1:] - img[:-1])) + np.sum(np.abs(img[:,1:] - img[:,:-1]))



def histogram_stretch(img_in, min_desired=0.0, max_desired=1.0, plot = True):
    """
    Stretches the histogram of an image 
    :param img_in: Input image
    :param min_desired: Goal minimum value
    :param max_desired: Goal maximum value
    :param plot: Bool; whether to plot the histogram
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()

    r = (max_desired - min_desired) / (max_val - min_val)

    img_out = r * (img_float - min_val) + min_desired
    #img_out = img_as_ubyte(img_out)
    if plot:
        h = plt.hist(img_in.ravel(), bins=255)
        h = plt.hist(img_out.ravel(), bins=255)
    return img_out

def threshold(img, tmin="otsu", tmax=255):
    """
    Creates binary image
    :param img: Image to threshold
    :param tmin: Minimum threshold val, default is "otsu".
    :param tmax: Maximum threshold val, default is 255. Can also be "otsu".
    :return: Binary image 
    """
    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    if tmin == "otsu":
        tmin = threshold_otsu(img)
    if tmax == "otsu":
        tmax = threshold_otsu(img)
    bin_img = (img >= tmin) & (img <= tmax)
    return bin_img

def edge_detection(img, filter_type="Median", footprint_size=10):
    """
    Performs edge-detection, by first applying a filter.
    Then applying prewitt.
    Finally, thresholding with otsus method.
    :param img: Image to find edges of
    :param filter_type: type of filter to apply. Either median, gaussian (/gauss/mean).
    :param footprint_size: size of filter kernel.
    :return: Binary image
    """
    if len(img.shape) == 3:
        img = color.rgb2gray(img)
    
    if filter_type.lower() == "median":
        img = median(img, np.ones([footprint_size,footprint_size]))
    elif filter_type.lower() in ["gauss", "gaussian", "mean"]:
        img = gaussian(img, footprint_size)
    else:
        print(":(")
        return
    
    im_edge = prewitt(img)
    im_out = threshold(im_edge)
    return im_out

def compute_outline(bin_img):
    """
    Computes the outline of a binary image
    """
    footprint = disk(1)
    dilated = dilation(bin_img, footprint)
    outline = np.logical_xor(dilated, bin_img)
    return outline


"""
idk wth this is lol
"""

def roiVals(img, roi):
    """
    Given an image and a mask; computes mean, standard deviation, and values within mask.
    """
    mask = roi > 0
    values = img[mask]
    mu = np.mean(values)
    std = np.std(values)
    return values, mu, std

def rois(img, rois, labels, min_hu = -200, max_hu = 1000, step_size = 1.0, plot = False):
    """
    Given an image, and a list of masks, and their labels;
    - Returns dictionary of pixelvalues in each mask, as well as mean and std.
    - Plots pdf of normal distributions for pixelvalues of each mask
    """
    roiDict = {}
    # Hounsfield unit limits of the plot
    hu_range = np.arange(min_hu, max_hu, step_size)
    for label, roi in zip(labels,rois):
        values, mu, std = roiVals(img, roi)
        roiDict[label] = {"mu" : mu, "std" : std, "values" : values}
        if plot:
            pdf = norm.pdf(hu_range, mu, std)
            plt.plot(hu_range, pdf, label=label)
            plt.legend()
    return roiDict

def labeller(val, roiDict, labels):
    return labels[np.argmax([norm.pdf(val, roiDict[i]["mu"] , roiDict[i]["std"]) for i in labels])]
    
def find_class_ranges(roiDict, labels, min=-200, max=1000,stepsize=1):
    label = labeller(-200, roiDict, labels)
    classRanges = []
    for val in np.arange(min, max, stepsize):
        nlabel = labeller(val, roiDict, labels)
        if nlabel != label:
            classRanges.append(val)
        label = nlabel
    return classRanges

def find_from_area(img, t_1, t_2, min_area, max_area, disk_size=5):
	estimate = (img > t_1) & (img < t_2)
	
	footprint = disk(disk_size)
	closed = binary_closing(estimate, footprint)
	opened = binary_opening(closed, footprint)
	
	label_img = measure.label(opened)
	region_props = measure.regionprops(label_img)

	label_img_filter = label_img.copy()
	for region in region_props:
		# Find the areas that do not fit our criteria
		if region.area > max_area or region.area < min_area:
			# set the pixels in the invalid areas to background
			for cords in region.coords:
				label_img_filter[cords[0], cords[1]] = 0
	i_area = label_img_filter > 0
	show_comparison(img, i_area, 'Found BLOB based on area')
	return


def gamma_map(img, gamma):
    """
    Applies gamma mapping on image pixel values
    :param img: Input image
    :param gamma: gamma
    :return: Image with gamma mapping applied
    """
    img_gam = img_as_float(img)**gamma
    if img_gam.dtype != img.dtype:
        return img_as_ubyte(img_gam)
    return img_gam

def RLE_encoder(img):
    """
    probably wrong lol
    """
    img = img.flatten()
    encoded = []
    prev = img[0]
    count = 0
    for i in img[1:]:
        count += 1
        if i != prev:
            encoded.append((prev, count))
            count = 0
        prev = i
    count += 1
    encoded.append((prev, count))

    return encoded

def RLE_decoder(encoded, shape=None):
    """
    probably wrong lol
    """
    img = []
    for num, val in encoded:
        img += [val] * num
    if shape != None:
        img = np.array(img).reshape(shape)
    return img
            

def chain_coder(startpixel, seq):
    """
    probably wrong lol
    """
    instructions = {
        0 : np.array([0,1]),
        1 : np.array([-1,1]),
        2 : np.array([-1,0]),
        3 : np.array([-1,-1]),
        4 : np.array([0,-1]),
        5 : np.array([1,-1]),
        6 : np.array([1,0]),
        7 : np.array([1,1]),
    }
    pixels = [np.array(startpixel)]
    for i in seq:
        pixels.append(pixels[-1] + instructions[i])
        if pixels[-1][0] < 0 or pixels[-1][1] < 0:
            print("Error : Negative pixel coords at ", len(pixels))
            return 
    pixels = pixels
    img = np.zeros((np.max(pixels, axis=0)+1))
    for x,y in pixels:
        img[x,y] = 1
    return img

def landmark_alignment_error(src, dst):
    e_x = src[:, 0] - dst[:, 0]
    error_x = np.dot(e_x, e_x)
    e_y = src[:, 1] - dst[:, 1]
    error_y = np.dot(e_y, e_y)
    f = error_x + error_y
    return f

def accumulator_img(img):
    accumulator_img = np.zeros(img.shape)
    accumulator_img[0,:] = img[0,:]

    for r in range(1,img.shape[0]):
        accumulator_img[r,0] = img[r,0] + min(accumulator_img[r-1,0], accumulator_img[r-1,1])
        for c in range(1,img.shape[1]-1):
            accumulator_img[r,c] = img[r,c] + min(accumulator_img[r-1,c-1],accumulator_img[r-1,c], accumulator_img[r-1,c+1])
        accumulator_img[r,-1] = img[r,-1] + min(accumulator_img[r-1,-1], accumulator_img[r-1,-2])
    return accumulator_img

def landmark_registration(src, dst, mode="Euclidean", src_img = None):
    print("Landmark alignment error before ", landmark_alignment_error(src, dst))
    if mode.lower() == "euclidean":
        tform = EuclideanTransform()
    elif mode.lower() == "affine":
        tform = AffineTransform()
    else:
        tform = SimilarityTransform()
    tform.estimate(src, dst)
    src_transform = matrix_transform(src, tform.params)
    print("Landmark alignment error after ", landmark_alignment_error(src_transform, dst))

    if src_img is not None:
        warped = warp(src_img, tform.inverse)
        warped = img_as_ubyte(warped)
        io.imshow(warped)
        return warped, src_transform, tform
    return src_transform, tform


def blobber(bin_img, min_area=None, max_area=None, min_perimeter=None, max_perimeter=None, min_circ=None, max_circ=None):
    label_img = measure.label(bin_img)
    fig, ax = plt.subplots(1,2)

    ax[0].imshow(label2rgb(label_img))

    region_props = measure.regionprops(label_img)
    n = len(region_props)
    areas = np.array([region.area for region in region_props])
    perimeters = np.array([region.perimeter for region in region_props])
    circ = areas * 4 * np.pi / perimeters**2
    
    attrBefore = {
    "areas": {"min" : min_area, "max": max_area, "vals" : areas}, 
    "pers": {"min" : min_perimeter, "max": max_perimeter, "vals" : perimeters}, 
    "circ": {"min" : min_circ, "max": max_circ, "vals" : circ}, 
    }
    keep = np.ones(n)
    for att in attrBefore:
        for i,val in enumerate(attrBefore[att]["vals"]):
            if not keep[i]:
                continue

            if attrBefore[att]["min"] != None:
                if val < attrBefore[att]["min"]:
                    keep[i] = 0
                    continue
            if attrBefore[att]["max"] != None:
                if val > attrBefore[att]["max"]:
                    keep[i] = 0
                    continue
    # Create a copy of the label_img
    label_img_filter = label_img
    for i in range(n):
        # Find the areas that do not fit our criteria
        if not keep[i]:
            # set the pixels in the invalid areas to background
            for cords in region_props[i].coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    ax[1].imshow(label2rgb(label_img_filter))
    region_props = measure.regionprops(label_img_filter)
    return attrBefore, region_props, label_img_filter



def play_vid(vid):
    n = vid.shape[2]
    for i in range(n):
        cv.imshow("img",vid[:,:,i])
        if cv.waitKey(10) == ord('q'):
            break
    cv.destroyAllWindows()
    return

def cmg(cfrom, cto): 
        from matplotlib.colors import LinearSegmentedColormap
        colors = [cfrom, cto] # first color is black, last is red
        cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=25)
        return cm

def show(img, mode="Default", cmap=None):
    """
    Shows image.
    :param img: image to show
    :param mode: Mode to show img in; options: Default, hsv, rgb, grey
    :param cmap: cmap to use when showing with default mode.
    """
    if mode=="Default":
        io.imshow(img, cmap=cmap)
    if mode.lower()=="hsv":
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(color.rgb2hsv(img)[:,:,0], cmap="hsv")
        ax[1].imshow(color.rgb2hsv(img)[:,:,1], cmap="RdGy")
        ax[2].imshow(color.rgb2hsv(img)[:,:,2], cmap="binary_r")
    if mode.lower()=="rgb":
        fig,ax = plt.subplots(1,3)
        ax[0].imshow(img[:,:,0], cmap=cmg((0,0,0),(1,0,0)))
        ax[1].imshow(img[:,:,1], cmap=cmg((0,0,0),(0,1,0)))
        ax[2].imshow(img[:,:,2], cmap=cmg((0,0,0),(0,0,1)))
    if mode.lower() in ["gray", "grey"]:
        io.imshow(color.rgb2gray(img))

def show_imgs(imgs):
    fig, ax = plt.subplots(1, len(imgs))
    for i,img in enumerate(imgs):
        ax[i].imshow(img)

def show_gallery(ims, cms="binary_r", n=4):
    n = np.min([len(ims), n])
    m = ceil(len(ims)/n)
    fig, ax = plt.subplots(m,n)
    if type(cms) == str:
        cms = [cms] * n * m
    if m > 1:
        for i, im in enumerate(ims):
            if im.dtype == float or im.dtype == bool:
                vmin, vmax = 0, 1
            else:
                vmin, vmax = 0, 255
            ax[int(i/n),i%n].imshow(im, vmin=vmin, vmax=vmax, cmap=cms[i])
    else:
        for i, im in enumerate(ims ):
            if im.dtype == float or im.dtype == bool:
                vmin, vmax = 0, 1
            else:
                vmin, vmax = 0, 255
            ax[i].imshow(im, vmin=vmin, vmax=vmax, cmap=cms[i])
    plt.show()