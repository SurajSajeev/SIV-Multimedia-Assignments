import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy import misc
from scipy import fftpack
pylab.rcParams['figure.figsize'] = (20.0, 7.0)
f=plt.figure()
im=cv2.imread('images.jpeg')


#1. reading the image and converting to greyscale 


im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
plt.imshow(im.astype('uint8'),cmap='gray')


#modules for dct and inverse dct
def dct2(a):
    return fftpack.dct( fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return fftpack.idct( fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')


#modules for dft and inverse dft
def dft2(a):
    return np.fft.fft2(a)

def idft2(a):
    return np.fft.ifft2(a)

"""This Is the DCT Part"""

#2.division of image blocks into 8x8 subblocks and 3.doing transformation
imsize = im.shape
dct = np.zeros(imsize)
for i in sp.r_[:imsize[0]:8]:
    for j in sp.r_[:imsize[1]:8]:
        dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)] )

#4. Quantization using thresholding
thresh = 0.012
dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))

percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)

print("Keeping only %f%% of the DCT coefficients" % (percent_nonzeros*100.0))

#applying inverse dct
im_dct = np.zeros(imsize)

for i in sp.r_[:imsize[0]:8]:
    for j in sp.r_[:imsize[1]:8]:
        im_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )


"""This is the DFT Part"""

dft = np.zeros(imsize,dtype='complex')
im_dft = np.zeros(imsize,dtype='complex')

# 8x8 DFT
for i in sp.r_[:imsize[0]:8]:
    for j in sp.r_[:imsize[1]:8]:
        dft[i:(i+8),j:(j+8)] = dft2( im[i:(i+8),j:(j+8)] )

# Thresh
thresh = 0.013
dft_thresh = dft * (abs(dft) > (thresh*np.max(abs(dft))))

# 8x8 iDFT
for i in sp.r_[:imsize[0]:8]:
    for j in sp.r_[:imsize[1]:8]:
        im_dft[i:(i+8),j:(j+8)] = idft2( dft_thresh[i:(i+8),j:(j+8)] )
        
percent_nonzeros_dft = np.sum( dft_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)
print("Keeping only %f%% of the DCT coefficients" % (percent_nonzeros*100.0))
print("Keeping only %f%% of the DFT coefficients" % (percent_nonzeros_dft*100.0))  
mse = (np.square(im - im_dct)).mean(axis=None)
mse2 = (np.square(im - im_dft.astype('uint8'))).mean(axis=None)
print("The mean square error of the DCT and DFT are as follows:"+" "+str(mse),mse2)
plt.figure()
plt.imshow( np.hstack( (im.astype('uint8'), im_dct.astype('uint8'), abs(im_dft.astype('uint8'))) ) ,cmap='gray')
plt.show()
plt.title("Comparison between original, DCT compressed and DFT compressed images" )