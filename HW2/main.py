import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

def dct_2d(image):
    M, N = image.shape
    dct = np.zeros((M, N))
    for u in range(M):
        for v in range(N):
            sum = 0
            for x in range(M):
                for y in range(N):
                    sum += image[x, y] * np.cos((2*x+1)*u*np.pi/(2*M)) * np.cos((2*y+1)*v*np.pi/(2*N))
            cu = 1/np.sqrt(2) if u == 0 else 1
            cv = 1/np.sqrt(2) if v == 0 else 1
            dct[u, v] = 2 * cu * cv * sum / np.sqrt(M*N)
    return dct

def idct_2d(dct):
    M, N = dct.shape
    image = np.zeros((M, N))
    for x in range(M):
        for y in range(N):
            sum = 0
            for u in range(M):
                for v in range(N):
                    cu = 1/np.sqrt(2) if u == 0 else 1
                    cv = 1/np.sqrt(2) if v == 0 else 1
                    sum += cu * cv * dct[u, v] * np.cos((2*x+1)*u*np.pi/(2*M)) * np.cos((2*y+1)*v*np.pi/(2*N))
            image[x, y] = 2 * sum / np.sqrt(M*N)
    return image

def dct_1d(vector):
    N = len(vector)
    dct = np.zeros(N)
    for k in range(N):
        sum = 0
        for n in range(N):
            sum += vector[n] * np.cos((2*n+1)*k*np.pi/(2*N))
        ck = 1/np.sqrt(2) if k == 0 else 1
        dct[k] = ck * sum * np.sqrt(2/N)
    return dct

def dct_2d_using_1d(image):
    M, N = image.shape
    temp = np.zeros((M, N))
    result = np.zeros((M, N))
    
    # Apply 1D-DCT to rows
    for i in range(M):
        temp[i] = dct_1d(image[i])
    
    # Apply 1D-DCT to columns
    for j in range(N):
        result[:, j] = dct_1d(temp[:, j])
    
    return result

def visualize_dct(dct_coeffs, title, filename):
    log_coeffs = np.log(np.abs(dct_coeffs) + 1)
    plt.figure(figsize=(10, 8))
    plt.imshow(log_coeffs, cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.savefig(filename)
    plt.close()

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def reconstruct_and_evaluate(dct_coeffs, original_image, method):
    reconstructed = idct_2d(dct_coeffs)
    cv2.imwrite(f'output/reconstructed_{method}.png', reconstructed)
    psnr = calculate_psnr(original_image, reconstructed)
    return reconstructed, psnr

def process_image(image, dct_function, method):
    start_time = time.time()
    dct_coeffs = dct_function(image)
    processing_time = time.time() - start_time
    
    visualize_dct(dct_coeffs, f'{method}-DCT Coefficients', f'output/{method}_dct_coeffs.png')
    reconstructed, psnr = reconstruct_and_evaluate(dct_coeffs, image, method)
    
    return dct_coeffs, reconstructed, psnr, processing_time

def main():
    if not os.path.exists('output'):
        os.makedirs('output')

    image_path = 'lena.png'
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image from '{image_path}'.")
    

    # 2D-DCT
    _, _, psnr_2d, time_2d_dct = process_image(image, dct_2d, '2d')
    
    # Two 1D-DCT
    _, _, psnr_1d, time_1d_dct = process_image(image, dct_2d_using_1d, '1d')
    
    # Print results
    print(f"PSNR for 2D-DCT: {psnr_2d:.2f} dB")
    print(f"PSNR for 1D-DCT: {psnr_1d:.2f} dB")
    print(f"Runtime for 2D-DCT: {time_2d_dct:.4f} seconds")
    print(f"Runtime for two 1D-DCT: {time_1d_dct:.4f} seconds")

if __name__ == "__main__":
    main()