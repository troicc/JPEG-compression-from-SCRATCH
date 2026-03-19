import numpy as np
import cv2
from collections import defaultdict
from base import (rgb_to_y, pad_image, dct2, zigzag_scan,
                    get_category, get_extra_bits, rle_one_block,
                    build_huffman_tree, extract_codes,
                    save_compressed)

def count_frequencies(all_zigzag_blocks):
    freq_dc = defaultdict(int)    # separate table for DC
    freq_ac = defaultdict(int)    # separate table for AC

    prev_dc = 0

    for zigzag in all_zigzag_blocks:       # zigzag is one row, shape (64,)
        dc, pairs = rle_one_block(zigzag)  # ← use your actual function name

        # DC: delta from previous block
        dc_delta = dc - prev_dc
        prev_dc  = dc

        dc_cat = get_category(abs(dc_delta))
        freq_dc[dc_cat] += 1               # DC symbol is just the category

        for (run, val) in pairs:
            if run == 0 and val == 0:
                freq_ac['EOB'] += 1
            else:
                cat = get_category(abs(val))
                freq_ac[(run, cat)] += 1
    return freq_dc, freq_ac

def encode_all_blocks(all_zigzag_blocks, codebook_dc, codebook_ac):
    bitstream = ""
    prev_dc = 0

    for zigzag in all_zigzag_blocks:
        dc, pairs = rle_one_block(zigzag)

        # ── DC ──
        dc_delta = dc - prev_dc
        prev_dc  = dc

        dc_cat, dc_extra = get_extra_bits(dc_delta)

        bitstream += codebook_dc[dc_cat]              # Huffman code for DC category
        if dc_cat > 0:
            bitstream += format(dc_extra, f'0{dc_cat}b')  # raw value bits

        # ── AC ──
        for (run, val) in pairs:
            if run == 0 and val == 0:
                bitstream += codebook_ac['EOB']
            else:
                cat, extra = get_extra_bits(val)
                bitstream += codebook_ac[(run, cat)]
                if cat > 0:
                    bitstream += format(extra, f'0{cat}b')

    return bitstream

def huffman_encode(zigzag_array):
    freq_dc, freq_ac = count_frequencies(zigzag_array)

    root_dc = build_huffman_tree(freq_dc)
    root_ac = build_huffman_tree(freq_ac)

    codebook_dc = extract_codes(root_dc)
    codebook_ac = extract_codes(root_ac)

    bitstream = encode_all_blocks(zigzag_array, codebook_dc, codebook_ac)

    return codebook_dc, codebook_ac, bitstream

def myencoder_final(image_path, quantization_table, output_path="compressed.bin"):
    
    
    color_image = cv2.imread(image_path)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    y_channel = rgb_to_y(color_image)
    original_height, original_width = y_channel.shape

    
    block_size = 8

    padded = pad_image(y_channel)
    padded_height, padded_width = padded.shape
    N=(padded_height * padded_width)//64
    zigzag_array = np.zeros((N, 64), dtype=np.int32)
    m=0

    all_dct_coefficients=np.zeros_like(padded, dtype=np.float32)
    all_quantized_coefficients = np.zeros_like(padded, dtype=np.float32)

    for i in range(0, padded_height, block_size):
        for j in range(0, padded_width, block_size):
            block = padded[i:i + block_size, j:j + block_size]
            block = block.astype(np.float32) 
            
            dct_coefficients = dct2(block)
            all_dct_coefficients[i:(i+block_size), j:(j+block_size)] = dct_coefficients

            
            quantized_coefficients = np.round(dct_coefficients / quantization_table).astype(np.int32)
            
            all_quantized_coefficients[i:(i+block_size), j:(j+block_size)] = quantized_coefficients
            
            zigzag_array[m,:]= zigzag_scan(quantized_coefficients)
            
            m=m+1

    zero_ratio = np.sum(zigzag_array == 0) / zigzag_array.size
    
    codebook_dc, codebook_ac, bitstream = huffman_encode(zigzag_array)
    
    save_compressed(
        output_path,
        codebook_dc, codebook_ac,
        bitstream,
        original_height, original_width
    )

    print(f"Encoded {image_path} → {output_path}")


if __name__ == '__main__':
    quantization_table = np.array([
                    [16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]
                ])

myencoder_final("ashley.jpg", quantization_table, "ashley.bin")