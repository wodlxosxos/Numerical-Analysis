import numpy as np
import cv2
from matplotlib import pyplot as plt
import random

def get_avg(fabric):
    mag_avg = np.zeros((64,64))
    for n in range(0, 10):
        row = random.randint(0, len(fabric) - 64 - 1)
        col = random.randint(0, len(fabric[0]) - 64 - 1)
        fabric_block = fabric[row:row+64, col:col+64]
        fabric_fft = np.fft.fft2(fabric_block)
        fabric_shift = np.fft.fftshift(fabric_fft)

        magnitude = np.log(np.abs(fabric_shift))
        mag_avg = mag_avg + magnitude
    mag_avg = mag_avg / 10
    return mag_avg

def find_topten(block):
    tmp = np.sort(block.reshape((1,32*32)))
    top = np.where(block >= tmp[0][len(tmp[0]) - 10])
    topten = np.zeros((len(top[0]), 2))
    topten[:,0] = np.where(block >= tmp[0][len(tmp[0]) - 10])[0]
    topten[:,1] = np.where(block >= tmp[0][len(tmp[0]) - 10])[1]
    return topten

def DFT_fun():
    comp_arr = np.zeros((20,32,32))
    n = 0
    while n < 20:
        comp_fabric = cv2.imread('DFTsample/fabric' + str(n+1) + '.jpg', 0)
        comp_fabric = get_avg(comp_fabric)
        comp_arr[n] = comp_fabric[0:32, 32:64]
        n += 1
    #img_idx = random.randint(1,20)
    #print(img_idx)
    for t in range(5):
        random_img = cv2.imread('DFTsample/fabricN.jpg', 0)
        print( '-' + str(t+1) + 'th try-')
        print('N')
        
        row = random.randint(0, len(random_img) - 64 - 1)
        col = random.randint(0, len(random_img[0]) - 64 - 1)
        fabric_block = random_img[row:row+64, col:col+64]
        fabric_fft = np.fft.fft2(fabric_block)
        fabric_shift = np.fft.fftshift(fabric_fft)
        magnitude = np.log(np.abs(fabric_shift))

        
        random_img = magnitude[0:32, 32:64]
        ran_top = find_topten(random_img)
        for m in range(20):
            top_ten = find_topten(comp_arr[m])
            num = 0
            for k in range(len(ran_top)):
                for i in range(len(top_ten)):
                    if ran_top[k][0] == top_ten[i][0] and ran_top[k][1] == top_ten[i][1]:
                        num += 1
                        break;
            if num >= len(ran_top) - 3:
                print(m + 1)
                break; 

if __name__ == "__main__":
    DFT_fun()
