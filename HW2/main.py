import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2

def load_image(image_path):
    """Load an image and convert it to grayscale."""
    image = Image.open(image_path).convert('L')
    return np.array(image)

def dct_2d(image):
    """Compute the 2D DCT of an image using numpy's optimized functions."""
    M, N = image.shape
    
    # Create the cosine transform matrix
    x = np.arange(M)
    y = np.arange(N)
    u = x.reshape(-1, 1)
    v = y.reshape(-1, 1)
    
    cos_u = np.sqrt(2 / M) * np.cos((2*x + 1) * u * np.pi / (2*M))
    cos_v = np.sqrt(2 / N) * np.cos((2*y + 1) * v * np.pi / (2*N))
    
    cos_u[0] /= np.sqrt(2)
    cos_v[0] /= np.sqrt(2)
    
    # Compute DCT using matrix multiplication
    dct_matrix = cos_u @ image @ cos_v.T
    
    return dct_matrix

def idct_2d(coefficients):
    """Compute the 2D IDCT of coefficients using numpy's optimized functions."""
    M, N = coefficients.shape
    
    # Create the cosine transform matrix
    x = np.arange(M)
    y = np.arange(N)
    u = x.reshape(-1, 1)
    v = y.reshape(-1, 1)
    
    cos_u = np.sqrt(2 / M) * np.cos((2*x + 1) * u * np.pi / (2*M))
    cos_v = np.sqrt(2 / N) * np.cos((2*y + 1) * v * np.pi / (2*N))
    
    cos_u[0] /= np.sqrt(2)
    cos_v[0] /= np.sqrt(2)
    
    # Compute IDCT using matrix multiplication
    reconstructed = cos_u.T @ coefficients @ cos_v
    
    return reconstructed

def dct_1d(signal):
    """Compute the 1D DCT of a signal using numpy's optimized functions."""
    N = len(signal)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    
    M = np.sqrt(2 / N) * np.cos((2*n + 1) * k * np.pi / (2*N))
    M[0] /= np.sqrt(2)
    
    return M @ signal

def idct_1d(coefficients):
    """Compute the 1D IDCT of coefficients using numpy's optimized functions."""
    N = len(coefficients)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    
    M = np.sqrt(2 / N) * np.cos((2*n + 1) * k * np.pi / (2*N))
    M[0] /= np.sqrt(2)
    
    return M.T @ coefficients

def dct_2d_using_1d(image):
    """Compute 2D DCT using two 1D DCTs."""
    return dct_1d(dct_1d(image.T).T)

def idct_2d_using_1d(coefficients):
    """Compute 2D IDCT using two 1D IDCTs."""
    return idct_1d(idct_1d(coefficients.T).T)

def cv2_dct_2d(image):
    """Compute the 2D DCT using OpenCV."""
    return cv2.dct(np.float32(image))

def cv2_idct_2d(coefficients):
    """Compute the 2D IDCT using OpenCV."""
    return cv2.idct(coefficients)

def psnr(original, reconstructed):
    """Calculate the PSNR between original and reconstructed images."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def visualize_and_save_dct_coefficients(coefficients, output_path):
    """Visualize the DCT coefficients in the log domain and save the image."""
    log_coefficients = np.log1p(np.abs(coefficients))
    plt.figure(figsize=(10, 8))
    plt.imshow(log_coefficients, cmap='viridis')
    plt.colorbar(label='Log magnitude')
    plt.title("DCT Coefficients (Log Domain)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_image(image, path):
    """Save the image to the specified path."""
    Image.fromarray(np.uint8(np.clip(image, 0, 255))).save(path)

def process_image(image, dct_function, idct_function, method_name):
    """Process an image using the given DCT and IDCT functions."""
    start_time = time.time()
    dct_coefficients = dct_function(image)
    reconstructed_image = idct_function(dct_coefficients)
    processing_time = time.time() - start_time
    
    psnr_value = psnr(image, reconstructed_image)
    
    visualize_and_save_dct_coefficients(dct_coefficients, os.path.join('output', f"dct_coefficients_{method_name}.png"))
    save_image(reconstructed_image, os.path.join('output', f"reconstructed_{method_name}.png"))
    
    return psnr_value, processing_time

def print_results(method_name, psnr_value, processing_time):
    """Print the results for a given method."""
    print(f"Results for {method_name}:")
    print(f"  PSNR: {psnr_value:.2f} dB")
    print(f"  Processing time: {processing_time:.4f} seconds")
    print()

def main():
    if not os.path.exists('output'):
        os.makedirs('output')

    # Load grayscale image
    image = load_image('lena.png')

    # Process image using 2D-DCT
    psnr_2d, time_2d = process_image(image, dct_2d, idct_2d, "2d")
    print_results("2D-DCT", psnr_2d, time_2d)

    # Process image using two 1D-DCT
    psnr_1d, time_1d = process_image(image, dct_2d_using_1d, idct_2d_using_1d, "1d")
    print_results("Two 1D-DCT", psnr_1d, time_1d)

    # Process image using OpenCV DCT (for comparison)
    psnr_cv2, time_cv2 = process_image(image, cv2_dct_2d, cv2_idct_2d, "cv2")
    print_results("OpenCV DCT", psnr_cv2, time_cv2)

if __name__ == "__main__":
    main()