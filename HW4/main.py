import numpy as np
import cv2
import os

# Quantization tables
quantization_table1 = np.array([
    [10, 7, 6, 10, 14, 24, 31, 37],
    [7, 7, 8, 11, 16, 35, 36, 33],
    [8, 8, 10, 14, 24, 34, 41, 34],
    [8, 10, 13, 17, 31, 52, 48, 37],
    [11, 13, 22, 34, 41, 65, 62, 46],
    [14, 21, 33, 38, 49, 62, 68, 55],
    [29, 38, 47, 52, 62, 73, 72, 61],
    [43, 55, 57, 59, 67, 60, 62, 59]
])

quantization_table2 = np.array([
    [10, 11, 14, 28, 59, 59, 59, 59],
    [11, 13, 16, 40, 59, 59, 59, 59],
    [14, 16, 34, 59, 59, 59, 59, 59],
    [28, 40, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59],
    [59, 59, 59, 59, 59, 59, 59, 59]
])

# Evaluate PSNR
def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel = 255.0
    return round(20 * np.log10(max_pixel / np.sqrt(mse)), 3)

def get_file_size(filename):
    return os.stat(filename).st_size

def dct_matrix_1d(N):
    mat = np.zeros((N, N))
    for k in range(N):
        coef = np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)
        for n in range(N):
            mat[k, n] = coef * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return mat

def dct_1d(data):
    M, N = data.shape
    DCT_mat_M, DCT_mat_N = dct_matrix_1d(M), dct_matrix_1d(N)
    return DCT_mat_M @ data @ DCT_mat_N.T

def idct_1d(data):
    M, N = data.shape
    DCT_mat_M, DCT_mat_N = dct_matrix_1d(M), dct_matrix_1d(N)
    return DCT_mat_N.T @ data @ DCT_mat_M

def run_length_encode(arr):
    values, counts, prev_val, count = [], [], arr[0], 1
    for elem in arr[1:]:
        if elem == prev_val:
            count += 1
        else:
            values.append(prev_val)
            counts.append(count)
            prev_val, count = elem, 1
    values.append(prev_val)
    counts.append(count)
    return np.array(values), np.array(counts)

def run_length_decode(values, counts):
    return np.repeat(values, counts)

def quantize(block, quant_table):
    return np.round(block / quant_table) * quant_table

def zigzag_indices(size):
    indices = [(i, j) for i in range(size) for j in range(size)]
    return sorted(indices, key=lambda x: (x[0] + x[1], -x[1] if (x[0] + x[1]) % 2 else x[1]))

def zigzag_scan(block):
    return np.array([block[i, j] for i, j in zigzag_indices(8)])

def de_zigzag_scan(zigzag):
    block, indices = np.zeros((8, 8)), zigzag_indices(8)
    for idx, (i, j) in enumerate(indices):
        block[i, j] = zigzag[idx]
    return block

def process_image_blocks(image, quant_table, block_size=8, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    h, w = image.shape
    processed_image, encoded = np.zeros_like(image, dtype=np.float32), [[0, 0]]
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):                
            block = np.float32(image[i:i+block_size, j:j+block_size])
            quantized_block = quantize(dct_1d(block), quant_table)
            zigzag, values, counts = zigzag_scan(quantized_block), *run_length_encode(zigzag_scan(quantized_block))
            encoded.extend(zip(counts, values))
            decoded_block = idct_1d(de_zigzag_scan(run_length_decode(values, counts)))
            processed_image[i:i+block_size, j:j+block_size] = decoded_block
    
    np.savez(os.path.join(output_dir, "encoded.npz"), np.array(encoded))
    cv2.imwrite(os.path.join(output_dir, "encoded.png"), np.array(encoded))
    return processed_image

if __name__ == "__main__":
    output_dir = "output"
    image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError("Image file not found.")
    
    height, width = image.shape
    image = image[:height - height % 8, :width - width % 8]

    for idx, quant_table in enumerate([quantization_table1, quantization_table2], 1):
        processed_image = process_image_blocks(image, quant_table, output_dir=f"{output_dir}/quant_table_{idx}")
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, f"quant_table_{idx}", "reconstructed_img.png"), processed_image)
        cv2.imwrite(os.path.join(output_dir, f"quant_table_{idx}", "lena_gray.png"), image)

        print(f"Quantization Table {idx}:")
        print("File size of original lena.png: ", get_file_size(os.path.join(output_dir, f"quant_table_{idx}", "lena_gray.png")), "bytes")
        print("File size of encoded file:", get_file_size(os.path.join(output_dir, f"quant_table_{idx}", "encoded.png")), "bytes")
        print("PSNR of lena.png and reconstructed_img:", psnr(processed_image, image))
        print()