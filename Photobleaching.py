#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 13:00:15 2021

@author: Mathew
"""

from skimage.io import imread
import matplotlib.pyplot as plt
from skimage import filters, measure
from skimage.filters import threshold_local
from PIL import Image
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


# Paths to analyse below


pathList = []

frame_rate=50/1000

root_path=(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/Rebuttal/Photobleaching/BIODIPY/All/")

pathList.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/Rebuttal/Photobleaching/BIODIPY/All/1/")
pathList.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/Rebuttal/Photobleaching/BIODIPY/All/2/")
pathList.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/Rebuttal/Photobleaching/BIODIPY/All/3/")
pathList.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/Rebuttal/Photobleaching/BIODIPY/All/4/")
pathList.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/Rebuttal/Photobleaching/BIODIPY/All/5/")
pathList.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/Rebuttal/Photobleaching/BIODIPY/All/6/")
pathList.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/Rebuttal/Photobleaching/BIODIPY/All/7/")
pathList.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/Rebuttal/Photobleaching/BIODIPY/All/8/")
pathList.append(r"/Users/Mathew/Documents/Manuscripts/ScotFluor Peptides/Rebuttal/Photobleaching/BIODIPY/All/9/")



filename = "Im.tif"

pixel_size = 103  # Pixel size in nm
photon_adu = 0.0265/0.96

# Function to load images:


def load_image(toload):

    image = imread(toload)

    return image


# Threshold image using otsu method and output the filtered image along with the threshold value applied:

def threshold_image_otsu(input_image):
    threshold_value = filters.threshold_otsu(input_image)
    binary_image = input_image > threshold_value

    return threshold_value, binary_image

def threshold_image_msd(input_image):
    threshold_value = input_image.mean()+2*input_image.std()
    binary_image = input_image > threshold_value

    return threshold_value, binary_image


def subtract_bg(input_image):
    block_size = 51
    local_thresh = threshold_local(input_image, block_size, method='gaussian')
    filtered = input_image - local_thresh

    return filtered
# Threshold image using otsu method and output the filtered image along with the threshold value applied:


def threshold_image_fixed(input_image, threshold_number):
    threshold_value = threshold_number
    binary_image = input_image > threshold_value

    return threshold_value, binary_image

# Label and count the features in the thresholded image:


def label_image(input_image):
    labelled_image = measure.label(input_image)
    number_of_features = labelled_image.max()

    return number_of_features, labelled_image

# Function to show the particular image:


def show(input_image, color=''):
    if(color == 'Red'):
        plt.imshow(input_image, cmap="Reds")
        plt.show()
    elif(color == 'Blue'):
        plt.imshow(input_image, cmap="Blues")
        plt.show()
    elif(color == 'Green'):
        plt.imshow(input_image, cmap="Greens")
        plt.show()
    else:
        plt.imshow(input_image)
        plt.show()


# Take a labelled image and the original image and measure intensities, sizes etc.
def analyse_labelled_image(labelled_image, original_image):
    measure_image = measure.regionprops_table(labelled_image, intensity_image=original_image, properties=(
        'area', 'perimeter', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length', 'mean_intensity', 'max_intensity'))
    measure_dataframe = pd.DataFrame.from_dict(measure_image)
    return measure_dataframe

# Curve fitting


def lifetime(x, y0, x0, A, tau):
    return y0 + A*np.exp(-(x-x0)/tau)

lifetime_all=[]

for i in range(len(pathList)):

    directory = pathList[i]+"/"

    toload = directory+filename

    imagestack = load_image(toload)

    zproj = np.mean(imagestack, axis=0)
    filt=zproj
    # filt = subtract_bg(zproj)
    im_threshold, im_binary = threshold_image_msd(filt)

    im = Image.fromarray(im_binary)

    im_number, im_labelled = label_image(im_binary)

    im_measurements = analyse_labelled_image(im_labelled, zproj)
    im_measurements.to_csv(directory + '/' + 'all_metrics.csv', sep='\t')

    # Need to extract frames
    number_of_clusters = im_number

    dimension_of_image = np.shape(imagestack)
    length = dimension_of_image[0]

    lifetimes=[]
    r_sq=[]
    
    for i in range(1,number_of_clusters):
        pix_to_check = im_labelled == i
        intensity = []
        time = []
        for j in range(0, length):
            frame_to_check = pix_to_check*imagestack[j, :, :]
            maximum = frame_to_check.max()
            intensity.append(maximum*photon_adu)
            time.append(j*frame_rate)
            
        init_vals = [min(intensity), 10*frame_rate,max(intensity),10*frame_rate]
        try:

            popt, _ = curve_fit(lifetime, time, intensity, p0=init_vals)

            
        except RuntimeError:
            print("Error - curve_fit failed")
        fit=lifetime(time,popt[0],popt[1],popt[2],popt[3])
        
        residuals = intensity-fit
        ss_res=np.sum(residuals**2)
        ss_tot = np.sum((intensity-np.mean(intensity))**2)
        r_squared = 1 - (ss_res / ss_tot)
        if(popt[3]<50):
            if(r_squared>0.7):
                plt.plot(time,fit)
                plt.plot(time,intensity)
                plt.xlabel('Time (s)')
                plt.ylabel('Intensity (ADU)')
                plt.savefig(directory+str(i)+'.pdf')
                plt.title('tau = '+str(round(popt[3],2))+' r_sq = '+str(round(r_squared,2)))
                plt.show()
            
                lifetimes.append(popt[3])
        
    lifetime_data=np.asarray(lifetimes)
    
    mean_lifetime=lifetime_data.mean()
    std_lifetime=lifetime_data.std()
    
    print(mean_lifetime)
    
    lifetime_all.append(mean_lifetime)
    
    
    df = pd.DataFrame(lifetime_all,columns =['Lifetime (s)'])
    df.to_csv(root_path+ 'PB_lifetime.csv', sep = '\t')
        
