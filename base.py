import numpy as np
import math
import struct
import json
from scipy.fftpack import dct, idct
from collections import defaultdict
import heapq

# ── image conversion ──
def rgb_to_y(image):
    # full-swing conversion using the BT.601 coefficients directly
    # input: RGB image uint8
    # output: Y channel float32 in range 0..255
    image_float = image.astype(np.float32)
    y = (0.299  * image_float[:,:,0] +
         0.587  * image_float[:,:,1] +
         0.114  * image_float[:,:,2])
    return y   # range 0..255, no headroom/footroom

# ── padding ──
def pad_image(image):
    
    height, width = image.shape

    padded_height = math.ceil(height / 8) * 8
    padded_width  = math.ceil(width  / 8) * 8

    # np.pad with 'edge' mode replicates the border pixels exactly
    pad_bottom = padded_height - height
    pad_right  = padded_width  - width

    padded = np.pad(
        image,
        pad_width=((0, pad_bottom), (0, pad_right)),
        mode='edge'     
    )

    return padded

# ── DCT ──
def dct2(block):
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(block):
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

# ── zigzag ──
index_order = sorted(
            ((x, y) for x in range(8) for y in range(8)),
            key=lambda p: (p[0]+p[1], -p[0] if (p[0]+p[1])%2==0 else p[0])
        )
def zigzag_scan(block):
    return [block[x, y] for x, y in index_order]

def zigzag_inverse(flat):
    block = np.zeros((8, 8), dtype=np.float32)
    for idx, (x, y) in enumerate(index_order):
        block[x, y] = flat[idx]
    return block

# ── RLE ──
def get_category(val):
    if val == 0:
        return 0
    return abs(val).bit_length()

def get_extra_bits(val):
    cat = get_category(abs(val))
    if val >= 0:
        return cat, val
    else:
        return cat, ((1 << cat) - 1) + val

def decode_value(bits_str, cat):
    raw = int(bits_str, 2)
    if bits_str[0] == '1':
        return raw
    else:
        return raw - ((1 << cat) - 1)

def rle_one_block(zigzag_row):
    dc = zigzag_row[0]
    pairs = []
    zero_run = 0
    for val in zigzag_row[1:]:
        if val == 0:
            zero_run += 1
        else:
            while zero_run > 15:
                pairs.append((15, 0))
                zero_run -= 16
            pairs.append((zero_run, val))
            zero_run = 0
    pairs.append((0, 0))
    return dc, pairs

# ── Huffman ──
class HNode:
    def __init__(self, symbol, freq, left=None, right=None):
        self.symbol = symbol
        self.freq   = freq
        self.left   = left
        self.right  = right

    def __lt__(self, other):          # heapq needs this for ordering
        return self.freq < other.freq

def build_huffman_tree(freq):
    heap = [HNode(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)                # arrange into min-heap

    while len(heap) > 1:
        left  = heapq.heappop(heap)    # lowest freq
        right = heapq.heappop(heap)    # second lowest

        merged = HNode(
            symbol = None,            # internal nodes have no symbol
            freq   = left.freq + right.freq,
            left   = left,
            right  = right
        )
        heapq.heappush(heap, merged)

    return heap[0]

def extract_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node.symbol is not None:       # leaf node
        codebook[node.symbol] = prefix if prefix else "0"
        return codebook
    extract_codes(node.left,  prefix + "0", codebook)
    extract_codes(node.right, prefix + "1", codebook)
    return codebook

def build_tree_from_codebook(codebook):
    """rebuild HNode tree from the codebook dict (symbol→bitstring)"""
    root = HNode(None, 0)
    for symbol, code in codebook.items():
        node = root
        for bit in code:
            if bit == '0':
                if node.left is None:
                    node.left = HNode(None, 0)
                node = node.left
            else:
                if node.right is None:
                    node.right = HNode(None, 0)
                node = node.right
        node.symbol = symbol
    return root

# ── file I/O ──
def bitstring_to_bytes(bitstring):
    remainder = len(bitstring) % 8
    padding = 0 if remainder == 0 else 8 - remainder
    bitstring += '0' * padding
    result = bytearray()
    for i in range(0, len(bitstring), 8):
        result.append(int(bitstring[i:i+8], 2))
    return bytes(result), padding

def save_compressed(filename, codebook_dc, codebook_ac,
                    bitstream, original_height, original_width):
    def codebook_to_json(cb):
        return {str(k): v for k, v in cb.items()}
    header = {
        'height': original_height,
        'width':  original_width,
        'dc':     codebook_to_json(codebook_dc),
        'ac':     codebook_to_json(codebook_ac),
    }
    header_bytes = json.dumps(header, separators=(',', ':')).encode('ascii')
    data_bytes, padding = bitstring_to_bytes(bitstream)
    with open(filename, 'wb') as f:
        f.write(b'MYJPEG')
        f.write(struct.pack('>I', len(header_bytes)))
        f.write(struct.pack('>I', padding))
        f.write(header_bytes)
        f.write(data_bytes)
    print(f"Saved to {filename}  ({len(data_bytes)/1024:.1f} KB)")

def load_compressed(filename):
    with open(filename, 'rb') as f:
        magic = f.read(6)
        if magic != b'MYJPEG':
            raise ValueError(f"Not a valid file: {magic}")
        header_len = struct.unpack('>I', f.read(4))[0]
        padding    = struct.unpack('>I', f.read(4))[0]
        header     = json.loads(f.read(header_len).decode('ascii'))
        data_bytes = f.read()

    bitstring = ''.join(format(b, '08b') for b in data_bytes)
    if padding > 0:
        bitstring = bitstring[:-padding]
        
    def json_to_codebook(cb_json):
        result = {}
        for k, v in cb_json.items():
            try:    parsed = eval(k)
            except: parsed = k
            result[parsed] = v
        return result
    return (json_to_codebook(header['dc']),
            json_to_codebook(header['ac']),
            bitstring,
            header['height'],
            header['width'])