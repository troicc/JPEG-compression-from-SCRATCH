# JPEG Compression from Scratch

A from-scratch implementation of JPEG-style image compression in Python.

## Pipeline

Image → RGB→Y → Pad → DCT → Quantize → Zigzag → RLE → Huffman → .bin
.bin  → Huffman decode → de-zigzag → Dequantize → IDCT → Crop → Image

## Usage

pip install -r requirements.txt

# encode
python encoder.py input.jpg output.bin

# decode
python decoder.py output.bin decoded.png

## File format (.bin)

[6B magic][4B header_len][4B padding][JSON codebooks][bitstream]
