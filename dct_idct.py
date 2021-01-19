import numpy as np
import cv2
import math


def sum_DCT(arr, Cu, Cv, u, v):
    result = np.zeros((1,3))
    for y in range(0,16):
        for x in range(0,16):
            result += arr[y][x] * math.cos(v*math.pi*((2*y + 1)/32))* math.cos(u*math.pi*((2*x + 1)/32))
    result = result * Cu * Cv
    result = result / 4
    return result


def sort_DCT(arr):
    for idx_height in range(0, int(len(arr)/16)):
        for idx_width in range(0, int(len(arr[0])/16)):
            tmp = arr[16*idx_height:16*idx_height + 16, 16*idx_width:16*idx_width + 16].reshape(1,256,3)
            th_R = np.sort(abs(tmp[:,:,0]))[0][240]
            th_G = np.sort(abs(tmp[:,:,1]))[0][240]
            th_B = np.sort(abs(tmp[:,:,2]))[0][240]
            for v in range(0, 16):
                for u in range(0, 16):
                    if abs(arr[16*idx_height + v][16*idx_width + u][0]) < th_R:
                        arr[16*idx_height + v][16*idx_width + u][0] = 0
                    if abs(arr[16*idx_height + v][16*idx_width + u][1]) < th_G:
                        arr[16*idx_height + v][16*idx_width + u][1] = 0
                    if abs(arr[16*idx_height + v][16*idx_width + u][2]) < th_B:
                        arr[16*idx_height + v][16*idx_width + u][2] = 0
    return arr
                    
def sum_IDCT(dct, y, x):
    result = np.zeros((1,3))
    for v in range(0, 16):
        Cv = 1
        if v == 0:
            Cv = 1 / math.sqrt(float(2))
        for u in range(0, 16):
            Cu = 1
            if u == 0:
                Cu = 1 / math.sqrt(float(2))
            result += dct[v][u] * Cu * Cv * math.cos(v*math.pi*((2*y + 1)/32))* math.cos(u*math.pi*((2*x + 1)/32))
    result = result / 4
    if result[0][0] < 0:
        result[0][0] = 0
    if result[0][0] > 255:
        result[0][0] = 255
    if result[0][1] < 0:
        result[0][1] = 0
    if result[0][1] > 255:
        result[0][1] = 255
    if result[0][2] < 0:
        result[0][2] = 0
    if result[0][2] > 255:
        result[0][2] = 255
    return result

def DCT_IDCT():
    rgb_image = cv2.imread('FDCTsample/testimage3.jpg')
    row = int(rgb_image.shape[0] / 16) * 16
    col = int(rgb_image.shape[1] / 16) * 16
    dct_output = np.zeros((row, col, 3))
    idct_output = np.zeros((row, col, 3))
    for idx_height in range(0, int(row/16)):
        for idx_width in range(0, int(col/16)):
            for v in range(0, 16):
                Cv = 1
                if v == 0:
                    Cv = 1 / math.sqrt(float(2))
                for u in range(0, 16):
                    Cu = 1
                    if u == 0:
                        Cu = 1 / math.sqrt(float(2))
                    dct_output[16 * idx_height + v][16 * idx_width + u] = sum_DCT(
                        rgb_image[16*idx_height:16*idx_height + 16, 16*idx_width:16*idx_width + 16], Cu, Cv, u, v)
    dct_output = sort_DCT(dct_output)
    
    for idx_height in range(0, int(row/16)):
        for idx_width in range(0, int(col/16)):
            for y in range(0, 16):
                for x in range(0, 16):
                    idct_output[16*idx_height + y][16*idx_width + x] = sum_IDCT(
                        dct_output[16*idx_height:16*idx_height + 16, 16*idx_width:16*idx_width + 16], y, x)
    cv2.imwrite('testresult3.png', idct_output)

if __name__ == "__main__":
    DCT_IDCT()
