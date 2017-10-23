## Script: Implementation of the Sobel Operator:

# Import necessary libraries: Using OpenCV only to read in the images/video frames:
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Used only to read in the image


# Function to perform convolution:
def convolve(input, kernel):

    # Allocate a matrix for the output:
    # This is where the error. If dtype is not specified, Python assigns it uint8. Thereby saturating below zero and above 255:
    # Thanks Srivatsan for pointing this out to me.
    output = np.zeros_like(input, dtype=np.float32)

    # First get all the parameters:
    [nRows, nCols] = input.shape

    kFull = kernel.shape[0]
    kHalf = (kernel.shape[0] -1)/2


    # Nested loops over each pixel in the image:
    for row in range(0,nRows):

        for col in range(0,nCols):

            sum =0
            kRow =0

            # Nested loops over flipped kernel:
            for i in range(row  + kHalf, row - kHalf - 1 , -1):

                kCol =0

                for j in range(col + kHalf, col - kHalf - 1, -1):

                    if (i < 0):
                        i =0

                    if (i>=nRows):
                        i = nRows - 1

                    if (j < 0):
                        j =0

                    if (j>=nCols):
                        j = nCols -1


                    # Multiply by each element of the kernel and add them all:
                    sum = sum + input[i][j]*kernel[kRow][kCol]

                    kCol+=1


                kRow+=1


            output[row][col] = sum


    # Normalize the output between 0 and 255: Remember the image can have negative values at this point
    #output = (output - np.mean(output))/(np.std(output))*127 + 127

    return output




# Read in the image from a file:
#input = Image.open('input.jpg').convert('L')
#input=mpimg.imread('shi-tomasi_test3.jpg')
input = cv2.imread('input.jpg', 0)

# Initialize the X and Y direction kernels:
# X:
kernel_x = np.array([ [1,0,-1] , [2,0,-2], [1,0,-1]])

# Y:
kernel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])


# # Convolve the image with the X-kerenel:
G_x_1= convolve(input, kernel_x)


# # Convolve the image with the Y-kerenel:
G_y_1 = convolve(input, kernel_y)


G= np.absolute(G_x_1) + np.absolute(G_y_1)

#G = np.sqrt(G_x_1**2 + G_y_1**2 , dtype=np.float32)   # Resultant magnitude of gradient at each pixel:


# Display using MATPLOTLIB:
plt.style.use('grayscale')
imgplot = plt.imshow(G)
plt.show()


# Display the image using OpenCV:
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',G_x)
# k = cv2.waitKey(0) & 0xFF
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
