import cv2
import numpy as np
import time
import os
from matplotlib import pyplot as plt

def psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel = 255.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val

def save_images(predicted_image, residual_image, block_size, search_range, method):
    output_dir = f'output/search_range_{search_range}'
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f'{output_dir}/pred_img_blocksize{block_size}_{method}.png', predicted_image)
    cv2.imwrite(f'{output_dir}/residual_image_blocksize{block_size}_{method}.png', residual_image)

def full_search(image1, image2, block_size, search_range):
    st = time.time()
    height, width = image1.shape
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32)
    predicted_image = np.zeros_like(image1)
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            ref_block = image2[i:i+block_size, j:j+block_size]
            best_match = np.inf
            m = [i, j]
            
            # Search for the best match in the search range
            for x in range(-search_range, search_range + 1):
                for y in range(-search_range, search_range + 1):
                    x_search, y_search = i + x, j + y
                    if x_search < 0 or y_search < 0 or x_search + block_size > height or y_search + block_size > width:
                        continue
                    candidate_block = image1[x_search:x_search+block_size, y_search:y_search+block_size]
                    error = np.mean((ref_block - candidate_block) ** 2)
                    if error < best_match:
                        best_match = error
                        dx, dy = x, y
                        m = [x_search, y_search]
            motion_vectors[i // block_size, j // block_size] = [dx, dy]
            predicted_image[i:i+block_size, j:j+block_size] = image1[m[0]:m[0]+block_size, m[1]:m[1]+block_size]

    residual_image = cv2.absdiff(image2, predicted_image)
    
    save_images(predicted_image, residual_image, block_size, search_range, method="full")
    
    PSNR = psnr(predicted_image, image2)
    et = time.time()
    time_cost = et - st
    return round(PSNR, 2), round(time_cost, 2)

def three_step_search(image1, image2, block_size, search_range):
    st = time.time()
    height, width = image1.shape
    motion_vectors = np.zeros((height // block_size, width // block_size, 2), dtype=np.int32)
    predicted_image = np.zeros_like(image1)
    k = int(np.log2(search_range))
    step_list = [2**i for i in range(k-1, -1, -1)]
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            ref_block = image2[i:i+block_size, j:j+block_size]
            local_search_range = search_range
            best_match = np.inf
            m = [i, j]
            for step in step_list:
                for x in range(-local_search_range, local_search_range + 1, step):
                    for y in range(-local_search_range, local_search_range + 1, step):
                        x_search, y_search = m[0] + x, m[1] + y
                        if x_search < 0 or y_search < 0 or x_search + block_size > height or y_search + block_size > width:
                            continue
                        candidate_block = image1[x_search:x_search+block_size, y_search:y_search+block_size]
                        error = np.mean((ref_block - candidate_block) ** 2)
                        if error < best_match:
                            best_match = error
                            m = [x_search, y_search]
                            dx, dy = x, y
                local_search_range //= 2
            motion_vectors[i // block_size, j // block_size] = [dx, dy]
            predicted_image[i:i+block_size, j:j+block_size] = image1[m[0]:m[0]+block_size, m[1]:m[1]+block_size]
    
    residual_image = cv2.absdiff(image2, predicted_image)

    save_images(predicted_image, residual_image, block_size, search_range, method="threestep")
    
    PSNR = psnr(predicted_image, image2)
    et = time.time()
    time_cost = et - st
    return round(PSNR, 2), round(time_cost, 2)

def run_experiment(image1, image2, block_size, search_range):
    print(f"########### Search Range +-{search_range} ############")
    
    psnr_full, time_cost_full = full_search(image1, image2, block_size, search_range)
    psnr_three_step, time_cost_three_step = three_step_search(image1, image2, block_size, search_range)
    
    print(f"Full Search:       PSNR: {psnr_full}, Runtime: {time_cost_full}")
    print(f"Three Step Search: PSNR: {psnr_three_step}, Runtime: {time_cost_three_step}")
    print()
    print(f"Full Search PSNR improvement: {psnr_full - psnr_three_step :.2f}")
    print(f"Three Step Search time saving: {time_cost_full - time_cost_three_step:.2f} seconds")
    print()
    
    return {
        'full_search': {'psnr': psnr_full, 'time': time_cost_full},
        'three_step_search': {'psnr': psnr_three_step, 'time': time_cost_three_step}
    }

def plot_results(results, block_size):
    search_ranges = list(results.keys())
    psnr_full = [results[range]['full_search']['psnr'] for range in search_ranges]
    psnr_three_step = [results[range]['three_step_search']['psnr'] for range in search_ranges]
    time_full = [results[range]['full_search']['time'] for range in search_ranges]
    time_three_step = [results[range]['three_step_search']['time'] for range in search_ranges]
    
    # 創建繪圖的文件夾
    plot_output_dir = 'output/plots'
    os.makedirs(plot_output_dir, exist_ok=True)
    
    # 繪製 PSNR 折線圖
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(search_ranges, psnr_full, marker='o', color='b', label='Full Search')
    plt.plot(search_ranges, psnr_three_step, marker='o', color='g', label='Three Step Search')
    
    for i, txt in enumerate(psnr_full):
        plt.text(search_ranges[i], psnr_full[i], f'{txt}', ha='center', va='bottom', fontsize=8)
    for i, txt in enumerate(psnr_three_step):
        plt.text(search_ranges[i], psnr_three_step[i], f'{txt}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Search Range')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs. Search Range')
    plt.legend()
    plt.grid(True)

    # 繪製 Runtime 折線圖
    plt.subplot(1, 2, 2)
    plt.plot(search_ranges, time_full, marker='o', color='b', label='Full Search')
    plt.plot(search_ranges, time_three_step, marker='o', color='g', label='Three Step Search')
    
    for i, txt in enumerate(time_full):
        plt.text(search_ranges[i], time_full[i], f'{txt}', ha='center', va='bottom', fontsize=8)
    for i, txt in enumerate(time_three_step):
        plt.text(search_ranges[i], time_three_step[i], f'{txt}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Search Range')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs. Search Range')
    plt.legend()
    plt.grid(True)

    # 調整布局並儲存圖片
    plt.tight_layout()
    plt.savefig(f'{plot_output_dir}/psnr_runtime_blocksize_{block_size}.png')

if __name__ == "__main__":
    path_to_first_image = 'one_gray.png'
    path_to_second_image = 'two_gray.png'
    image1 = cv2.imread(path_to_first_image, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(path_to_second_image, cv2.IMREAD_GRAYSCALE)
    
    block_size = 8
    search_ranges = [8, 16, 32]
    
    if not os.path.exists('output'):
        os.makedirs('output')
        
    results = {}
    for search_range in search_ranges:
        results[search_range] = run_experiment(image1, image2, block_size, search_range)
    
    plot_results(results, search_ranges)