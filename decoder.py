import numpy as np
import cv2
import math
from common import (idct2, zigzag_inverse, decode_value,
                    build_tree_from_codebook, load_compressed)

def mydecoder(bin_path, quantization_table, output_path):
    codebook_dc, codebook_ac, bitstream, orig_h, orig_w = load_compressed(bin_path)

    root_dc = build_tree_from_codebook(codebook_dc)
    root_ac = build_tree_from_codebook(codebook_ac)

    padded_h = math.ceil(orig_h / 8) * 8
    padded_w = math.ceil(orig_w / 8) * 8
    N = (padded_h * padded_w) // 64
    reconstructed = np.zeros((padded_h, padded_w), dtype=np.float32)

    i = 0
    prev_dc = 0

    for m in range(N):

            # ── decode DC ──
            node = root_dc
            while node.symbol is None:
                node = node.left if bitstream[i] == '0' else node.right
                i += 1

            dc_cat = node.symbol
            if dc_cat == 0:
                dc_delta = 0
            else:
                dc_bits = bitstream[i:i+dc_cat]
                i += dc_cat
                dc_delta = decode_value(dc_bits, dc_cat)

            dc = prev_dc + dc_delta
            prev_dc = dc

            # ── decode AC ──
            ac = [0] * 63
            ac_idx = 0

            while ac_idx < 63:
                node = root_ac
                while node.symbol is None:
                    node = node.left if bitstream[i] == '0' else node.right
                    i += 1

                sym = node.symbol
                if sym == 'EOB':
                    break                      # rest are zero, already initialised

                run, cat = sym
                if cat == 0:                   # ZRL (15,0) — skip 16 zeros
                    ac_idx += 16
                    continue

                extra_bits = bitstream[i:i+cat]
                i += cat
                val = decode_value(extra_bits, cat)

                ac_idx += run                  # skip zero run
                ac[ac_idx] = val
                ac_idx += 1

    zigzag_1d = zigzag_inverse([dc] + ac)
    block_spatial = idct2(zigzag_1d * quantization_table) + 128.0
    row = (m // (padded_w // 8)) * 8
    col = (m %  (padded_w // 8)) * 8
    reconstructed[row:row+8, col:col+8] = block_spatial

    reconstructed = reconstructed[:orig_h, :orig_w]
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, reconstructed)
    print(f"Decoded {bin_path} → {output_path}")

    return reconstructed