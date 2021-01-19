import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

def eigenface():
    #1 image 가져오기
    img_array = []
    mean = []
    num_of_image = 0
    for filename in os.listdir("my_faces"):
        fn = os.path.join(os.getcwd(), "my_faces//" + filename)
        stream = open(fn, "rb")
        bytes = bytearray(stream.read())
        nparray = np.asarray(bytes)
        img = cv2.imdecode(nparray, cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (1, 4096))
        img = img[0]
        img_array.append(img)
        
    """mean = mean/num_of_image
    img_array = img_array - mean"""
    img_array = np.array(img_array)
    img_array = np.transpose(img_array)
    for i in range(0, len(img_array)):
        mean.append(np.mean(img_array[i]))
        img_array[i] = img_array[i] - mean[i]
    img_array = np.transpose(img_array)

    mean = np.array(mean)
    mean = np.reshape(mean, (64,64))
    plt.imshow(mean, cmap = "gray");plt.show()
    
    
    #2 SVD적용
    """tmp_array = img_array.T
    u, s ,vt = np.linalg.svd(tmp_array)
    U = u.T

    coefficient = []
    for dirname in os.listdir("./for_test"):
        if dirname.find(".ini") == -1:
            for filename in os.listdir("./for_test/" + dirname):
                if filename.find("pgm") != -1:
                    fn = os.path.join(os.getcwd(), "for_test//" + dirname + "//"+ filename)
                    stream = open(fn, "rb")
                    bytes = bytearray(stream.read())
                    nparray = np.asarray(bytes)
                    img = cv2.imdecode(nparray, cv2.IMREAD_GRAYSCALE)
                    img = np.reshape(img, (1, 4096))
                    img = img[0]
                    for i in range(0, 4096):
                        img[i] = img[i] - mean[i]
                    tmp_co = []
                    for i in range(0, 40):
                        tmp_co.append(img.T@U[i])
                    coefficient.append(tmp_co)
    reimaging = [[]] * 50
    print(U[0]*3)
    print(np.array(U[0])*3)
    for j in range(0, 50):
        for i in range(0, 40):
            if len(reimaging[j]) == 0:
                reimaging[j] = np.array(U[i])*coefficient[j][i]
            else :
                reimaging[j] = reimaging[j] + np.array(U[i])*coefficient[j][i]
        for i in range(0,4096):
            reimaging[j][i] = reimaging[j][i] + mean[i]
    for j in range(0, 50):
        reimaging[j] = np.reshape(reimaging[j], (64,64))
    test_tmp = reimaging[45]
    test_tmp = np.vstack([test_tmp, reimaging[46]])
    test_tmp = np.vstack([test_tmp, reimaging[47]])
    test_tmp = np.vstack([test_tmp, reimaging[48]])
    test_tmp = np.vstack([test_tmp, reimaging[49]])
    plt.imshow(test_tmp, cmap = "gray");plt.show()"""

if __name__ == "__main__":
    eigenface()
