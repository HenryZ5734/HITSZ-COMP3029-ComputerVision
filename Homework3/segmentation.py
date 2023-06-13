import numpy as np
import random
import math
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
from skimage import color

### Clustering Methods for 1-D points
def kmeans(features, k, num_iters=500):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    print(features.shape)
    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for _ in range(num_iters):
        ### YOUR CODE HERE
        for i in range(N):
            assignments[i] = np.argmin(np.linalg.norm(features[i] - centers, axis=1))
        new_centers = np.array([features[assignments == i].mean(axis=0) for i in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
        ### END YOUR CODE

    return assignments

### Clustering Methods for colorful image
def kmeans_color(features, k, num_iters=100):
    H, W, D = features.shape
    N = H * W

    # Randomly initalize cluster centers
    features = features.reshape((N, D))
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)
    #Like the kmeans function above
    ### YOUR CODE HERE
    for _ in range(num_iters):
        for i in range(N):
            assignments[i] = np.argmin(np.linalg.norm(features[i] - centers, axis=1))
        new_centers = np.array([features[assignments == i].mean(axis=0) for i in range(k)])
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    ### END YOUR CODE

    return assignments.reshape(H, W)

#找每个点最后会收敛到的地方（peak）
def findpeak(data, idx, r):
    t = 0.01
    shift = np.array([1])
    data_point = data[:, idx]
    dataT = data.T
    data_pointT = data_point.T
    data_pointT = data_pointT.reshape(1, 3)

    # Runs until the shift is smaller than the set threshold
    while (shift > t).any() :
        ### YOUR CODE HERE
        # 计算当前点和所有点之间的距离
        dis = np.linalg.norm(dataT - data_pointT, axis=1)
        # 并筛选出在半径r内的点，计算mean vector（这里是最简单的均值，也可尝试高斯加权）
        data_intra = dataT[dis < r]
        data_point_new = np.mean(data_intra, axis=0)
        # 用新的center（peak）更新当前点，直到满足要求跳出循环
        shift = data_point_new - data_point
        data_point = data_point_new
        ### END YOUR CODE

    return data_point


# Mean shift algorithm
# 可以改写代码，鼓励自己的想法，但请保证输入输出与notebook一致
def meanshift(data, r):
    labels = np.zeros(len(data.T))
    peaks = [] #聚集的类中心
    label_no = 1 #当前label
    labels[0] = label_no

    # findpeak is called for the first index out of the loop
    peak = findpeak(data, 0, r)
    peaks.append(peak)

    # Every data point is iterated through
    ### YOUR CODE HERE
    for idx in range(1, len(data.T)):
        # 遍历数据，寻找当前点的peak
        peak = findpeak(data, idx, r)
        # 并实时关注当前peak是否会收敛到一个新的聚类（和已有peaks比较）
        dis = np.linalg.norm(np.array(peaks) - peak, axis=1)
        dis_min = np.min(dis)
        if dis_min >= r:
            # 若是，更新label_no，peaks，labels，继续
            label_no += 1
            labels[idx] = label_no
            peaks.append(peak)
        else:
            # 若不是，当前点就属于已有类，继续
            labels[idx] = np.argmin(dis) + 1
    ### END YOUR CODE
    #print(set(labels))
    return labels, np.array(peaks).T


# image segmentation
def segmIm(img, r):
    # Image gets reshaped to a 2D array
    img_reshaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # We will work now with CIELAB images
    imglab = color.rgb2lab(img_reshaped)
    # segmented_image is declared
    segmented_image = np.zeros((img_reshaped.shape[0], img_reshaped.shape[1]))


    labels, peaks = meanshift(imglab.T, r)
    # Labels are reshaped to only one column for easier handling
    labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

    # We iterate through every possible peak and its corresponding label
    for label in range(0, peaks.shape[1]):
        # Obtain indices for the current label in labels array
        inds = np.where(labels_reshaped == label + 1)[0]

        # The segmented image gets indexed peaks for the corresponding label
        corresponding_peak = peaks[:, label]
        segmented_image[inds, :] = corresponding_peak
    # The segmented image gets reshaped and turn back into RGB for display
    segmented_image = np.reshape(segmented_image, (img.shape[0], img.shape[1], 3))

    res_img=color.lab2rgb(segmented_image)
    res_img=color.rgb2gray(res_img)
    return res_img


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    accuracy1 = np.sum(np.abs(mask_gt - mask)) / mask_gt.size
    accuracy2 = np.sum(np.abs(mask_gt - (1 - mask))) / mask_gt.size
    accuracy = max(accuracy1, accuracy2)
    ### END YOUR CODE

    return accuracy

