import numpy as np
import matplotlib.pyplot as plt

def matching(test, point, distance):
    result = np.zeros((1,5))[0]
    for tmp_test in test:
        tmp_dist = 0
        sel_clus = -1
        for clus_idx in range(5):
            if sum((tmp_test-point[clus_idx])**2) <= distance[clus_idx] :
                if tmp_dist**2 < sum(tmp_test-point[clus_idx])**2 :
                    tmp_dist = sum(tmp_test-point[clus_idx])**2
                    sel_clus = clus_idx
        if sel_clus != -1:
            result[sel_clus] += 1
    return result

def k_mean_test():
    #cluster array
    point= np.zeros((5,3))
    distance = np.zeros((1,5))[0]
    #cluster1 : center estimate (0,0,0)
    point[0][0] = np.mean(np.random.normal(0, 2, 100))
    point[0][1] = np.mean(np.random.normal(0, 2, 100))
    point[0][2] = np.mean(np.random.normal(0, 1, 100))
    distance[0] = 4**2
    #cluster2 : center estimate (5,0,1)
    point[1][0] = np.mean(np.random.normal(7.5, 2, 100))
    point[1][1] = np.mean(np.random.normal(0, 1, 100))
    point[1][2] = np.mean(np.random.normal(1.5, 2, 100))
    distance[1] = 4**2
    #cluster3 : center estimate (7,1,6)
    point[2][0] = np.mean(np.random.normal(10.5, 3, 100))
    point[2][1] = np.mean(np.random.normal(1.5, 2, 100))
    point[2][2] = np.mean(np.random.normal(9, 2, 100))
    distance[2] = 6**2
    #cluster4 : center estimate (6,3,13)
    point[3][0] = np.mean(np.random.normal(9, 2, 100))
    point[3][1] = np.mean(np.random.normal(4.5, 2, 100))
    point[3][2] = np.mean(np.random.normal(19.5, 3, 100))
    distance[3] = 6**2
    #cluster5 : center estimate (-1,2,10)
    point[4][0] = np.mean(np.random.normal(-1.5, 4, 100))
    point[4][1] = np.mean(np.random.normal(3, 3, 100))
    point[4][2] = np.mean(np.random.normal(15, 2, 100))
    distance[4] = 8**2

    #--------------------------------------------
    #cluster 1~5 test
    test_point = np.zeros((5,100,3))
    #1
    test_point[0][:,0] = np.random.normal(0, 2, 100)
    test_point[0][:,1] = np.random.normal(0, 2, 100)
    test_point[0][:,2] = np.random.normal(0, 1, 100)
    #2
    test_point[1][:,0] = np.random.normal(7.5, 2, 100)
    test_point[1][:,1] = np.random.normal(0, 1, 100)
    test_point[1][:,2] = np.random.normal(1.5, 2, 100)
    #3
    test_point[2][:,0] = np.random.normal(10.5, 3, 100)
    test_point[2][:,1] = np.random.normal(1.5, 2, 100)
    test_point[2][:,2] = np.random.normal(9, 2, 100)
    #4
    test_point[3][:,0] = np.random.normal(9, 2, 100)
    test_point[3][:,1] = np.random.normal(4.5, 2, 100)
    test_point[3][:,2] = np.random.normal(19.5, 3, 100)
    #5
    test_point[4][:,0] = np.random.normal(-1.5, 4, 100)
    test_point[4][:,1] = np.random.normal(3, 3, 100)
    test_point[4][:,2] = np.random.normal(15, 2, 100)
    
    """fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.scatter(test_point[0][:,0], test_point[0][:,1], test_point[0][:,2])
    ax.scatter(test_point[1][:,0], test_point[1][:,1], test_point[1][:,2])
    ax.scatter(test_point[2][:,0], test_point[2][:,1], test_point[2][:,2])
    ax.scatter(test_point[3][:,0], test_point[3][:,1], test_point[3][:,2])
    ax.scatter(test_point[4][:,0], test_point[4][:,1], test_point[4][:,2])
    plt.show()
    return 0"""
    for idx in range(5):
        print("Cluster " + str(idx + 1) + " test")
        result = matching(test_point[idx], point, distance)
        print(result)

    #---------------------------------------
    #cluster6 for test
    test = np.zeros((100,3))
    test[:,0] = 2*np.random.randn(100) - 1.5
    test[:,1] = 1*np.random.randn(100) + 1.5
    test[:,2] = 1*np.random.randn(100) + 6
    print("Cluster None for test")
    result = matching(test, point, distance)
    result = 100 - result[0] - result[1] - result [2] - result[3] - result[4]
    print(int(result))
if __name__ == "__main__":
    k_mean_test()
