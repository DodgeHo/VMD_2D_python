# coding: utf-8
from VMD2D import VMD2D
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

# Replace 'VMD_2D' with your Python implementation of the VMD_2D function

# Read the image
try:
    image = cv2.imread('Sample.bmp')
    if image is None:
        raise FileNotFoundError("Could not load image file")
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

# Convert to grayscale if it's a color image
if image.ndim == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Parameters
alpha = 5000       # Bandwidth constraint
tau = 0.25         # Lagrangian multipliers dual ascent time step
K = 5              # Number of modes
DC = 1             # Includes DC part (first mode at DC)
init = 1           # Initialize omegas randomly
tol = K * 10 ** -6 # Tolerance (for convergence)
eps = 2.2204e-16

# Run VMD
u, u_hat, omega = VMD2D(image, alpha, tau, K, DC, init, tol, eps)

## Display 
num_modes = u.shape[2]
num_plots = num_modes + 2  # +2 for the input image and reconstructed composite

# Calculate the number of rows and columns for subplots
num_rows = 2
num_columns = (num_plots + 1) // 2  # Adjust columns based on the total number of plots

plt.figure(figsize=(15, 8))  # Adjust the figure size as needed

# Display input image
plt.subplot(num_rows, num_columns, 1)
plt.imshow(image, cmap='gray')
plt.title('Input image')
plt.axis('equal')
plt.axis('off')

# Display each mode
for k in range(num_modes):
    plt.subplot(num_rows, num_columns, k + 2)  # Adjust subplot index
    plt.imshow(u[:, :, k], cmap='gray')
    plt.title(f'Mode #{k + 1}')
    plt.axis('equal')
    plt.axis('off')

# Display reconstructed composite
composite_index = num_rows * num_columns  # Position of the last subplot
plt.subplot(num_rows, num_columns, composite_index)
plt.imshow(np.sum(u, axis=2), cmap='gray')
plt.title('Reconstructed composite')
plt.axis('equal')
plt.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.2)  # Adjust spacing: wspace for width, hspace for height
plt.tight_layout()
plt.show()





