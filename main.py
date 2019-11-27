# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 22:27:16 2019
@author: Metin Mert Akçay
"""
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import cv2
import sys
import os

MAXIMUM_ITERATION = 10
FOLDER_PATH = 'Images'
OUTPUT_PATH = 'Output'
GROUND_TRUTH_PATH = 'Ground_truth'
K = 100
M = 10


""" 
    This function is used for reading the image.
    @param image_name: name of the image
    @return image: image with pixel values
"""
def read_image(image_name):
    image = cv2.imread(os.path.join(FOLDER_PATH, image_name))
    b, g, r = cv2.split(image)
    # b,g,r image is converted to r,g,b.
    image = cv2.merge([r,g,b])
    return image


"""
    This function is for displaying the picture on the screen.
    @param image_name: name of the image
    @param image: image with pixel values
"""
def show_image(image_name, image):
    plt.imshow(np.array(image))
    plt.title(image_name)
    plt.show()    


""" 
    This function is used to write the image taken as parameter to the related file.
    @param image_name: name of the image to be saved
    @param folder_name: folder name to be saved
    @param image: image to be written
"""
def write_image(image_name, folder_path, image):
    if(not(os.path.exists(folder_path))):
        os.makedirs(folder_path)
    cv2.imwrite(os.path.join(folder_path , image_name), np.array(image))


"""
    This function is used to find the distance between two given points.
    @param point_1: first point which format is [r, g, b, x, y]
    @param point_2: second point which format is [r, g, b, x, y]
    @param S: Square root of the N divided by K
    @return distance between two points
"""
def distance_func(point_1, point_2, S):
    distance_rgb = (((int(point_1[0]) - int(point_2[0])) ** 2) + ((int(point_1[1]) - int(point_2[1])) ** 2) + ((int(point_1[2]) - int(point_2[2])) ** 2)) ** 0.5
    distance_xy = ((point_1[3] - point_2[3]) ** 2 + (point_1[4] - point_2[4]) ** 2) ** 0.5
    return distance_rgb + M / S * distance_xy


"""
    This function is used to convert r, g, b pixel to grayscale
    @param r, g, b: pixel value
    @return grayscale value
"""
def rgb_to_grayscale(r, g, b):
    return (0.3 * r) + (0.59 * g) + (0.11 * b);


"""
    This function is used to find local minimum gradient. (3 x 3)
    @param center: Mid point for S x S area
    @param image: image to be segmented
"""
def find_local_minimum_gradient(center, image):
    min_gradient = sys.maxsize
    local_min_point = center
    # search 3 x 3 area
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            r1, g1, b1 = image[i + 1][j]
            r2, g2, b2 = image[i - 1][j]
            r3, g3, b3 = image[i][j + 1]
            r4, g4, b4 = image[i][j - 1]
            p1 = rgb_to_grayscale(r1, g1, b1)
            p2 = rgb_to_grayscale(r2, g2, b2)
            p3 = rgb_to_grayscale(r3, g3, b3)
            p4 = rgb_to_grayscale(r4, g4, b4)
            if ((p1 - p2) ** 2) ** 0.5 + ((p3 - p4) ** 2) ** 0.5 < min_gradient:
                min_gradient = ((p1 - p2) ** 2) ** 0.5 + ((p3 - p4) ** 2) ** 0.5
                local_min_point = [i, j]
    return local_min_point


"""
    This function is used to create a list which hold points inside. (x, y coordinate)
    @param row - column: These values ​​are used to determine the size of the list.
"""
def create_clusters(row, column):
    clusters_row = []
    for i in range(row):
        clusters_column = []
        for j in range(column):
            clusters_column.append(0)
        clusters_row.append(clusters_column)
    return clusters_row


"""
    This function is used to create a list which keep distances.
    @param row - column: These values ​​are used to determine the size of the list.
"""
def create_distances(row, column):
    distances_row = []
    for i in range(row):
        distances_column = []
        for j in range(column):
            distances_column.append(sys.maxsize)
        distances_row.append(distances_column)
    return distances_row


"""
    This function is used to create a list which count point in clusters.
    @param size: size of list
"""
def create_number_of_points_in_cluster(size):
    number_of_points_in_cluster = []
    for i in range(size):
        number_of_points_in_cluster.append(0)
    return number_of_points_in_cluster


"""
    This function is used to clear the distance list.
    @param row - column: These values ​​are used to determine the size of the list.
    @param distances: it is a list which keep minimum distances.
"""
def clear_distance_list(row, column, distances):
    for i in range(row):
        for j in range(column):
            distances[i][j] = sys.maxsize


"""
    In this function, slic algorithm is executed. Over-segmentation was performed with using slic algorithm.
    @param image_name: name of the image
    @param image: image to be segmented
"""
def slic(image_name, image):
    row, column = image.shape[:2]
    
    N = row * column            # number of pixel in image
    S = int((N / K) ** 0.5)     # Approximately a super-pixel area (S x S)
    
    # initialize centers
    centers = []
    for i in range(int(S / 2), row - int(S / 2), S):
        for j in range(int(S / 2), column - int(S / 2), S):
            # find local minimum gradient
            local_point = find_local_minimum_gradient([i, j], image)
            r, g, b = image[local_point[0]][local_point[1]]
            center = [r, g, b, local_point[0],local_point[1]]
            centers.append(center)
    
    # clusters: it is a list which keeps closest center id.
    # distance: it is a list which keeps shortest distance between point and centers.
    # number_of_points_in_cluster: it is a list which holds the number of points in the cluster.
    clusters = create_clusters(row, column)
    distances = create_distances(row, column)
    number_of_points_in_cluster = create_number_of_points_in_cluster(len(centers))
    for i in range(MAXIMUM_ITERATION):
        clear_distance_list(row, column, distances)
        for j in range(len(centers)):
            # search 2S x 2S area
            for k in range(centers[j][3] - S, centers[j][3] + S): 
                for l in range(centers[j][4] - S, centers[j][4] + S):
                
                    if k >= 0 and k < row and l >= 0 and l < column:
                        r, g, b = image[k][l]
                        distance = distance_func(centers[j], [r, g, b, k, l], S)
                        
                        # find mininmum distance and assign a closest center
                        if distance < distances[k][l]:
                            distances[k][l] = distance
                            clusters[k][l] = j

        # x and y values ​​of the center points and the number of points in cluster reset.
        for j in range(len(centers)):
            number_of_points_in_cluster[j] = centers[j][3] = centers[j][4] = 0

        # calculate new x and y values of center point and the number of points in cluster
        for j in range(row):
            for k in range(column):
                center_id = clusters[j][k]
                centers[center_id][3] += j
                centers[center_id][4] += k
                number_of_points_in_cluster[center_id] += 1

        # assign new centers r,g,b values        
        for j in range(len(centers)):
            try:
                centers[j][3] = centers[j][3] // number_of_points_in_cluster[j]
                centers[j][4] = centers[j][4] // number_of_points_in_cluster[j]
            except ZeroDivisionError:
                centers[j][3] = centers[j][3] // 1
                centers[j][4] = centers[j][4] // 1
            centers[j][0], centers[j][1], centers[j][2] = image[centers[j][3]][centers[j][4]]
    
    
    # this part is used to enforce connectivity
    label = 0
    adjacent_label = 0
    limit = int(row * column / len(centers))
    new_clusters = -1 * np.ones(image.shape[:2]).astype(np.int64)
    # (-1, 0), (0, -1), (1, 0), (0, 1)
    dx_4 = [-1, 0, 1, 0]
    dy_4 = [0, -1, 0, 1]
    items = []
    for i in range(row):
        for j in range(column):
            items = []
            if new_clusters[i][j] == -1:
                items.append((i, j))
                for dx, dy in zip(dx_4, dy_4):
                    x = items[0][0] + dx
                    y = items[0][1] + dy
                    if (x >= 0 and x < row and y >= 0 and y < column and new_clusters[x][y] >= 0):
                        adjacent_label = new_clusters[x][y]
            
            item_count = len(items)
            k = 0
            while k < item_count:
                for dx, dy in zip(dx_4, dy_4):
                    x = items[k][0] + dx
                    y = items[k][1] + dy

                    if (x >= 0 and x < row and y >= 0 and y < column):
                        if new_clusters[x][y] == -1 and clusters[i][j] == clusters[x][y]:
                            items.append((x, y))
                            new_clusters[x][y] = label
                            item_count += 1
                k += 1
            
            # if the number of points in the cluster is less than the specified value, this cluster is combined with the neighboring cluster.
            if item_count <= (limit >> 2):
                for k in range(item_count):
                    x = items[k][0]
                    y = items[k][1]
                    new_clusters[x][y] = adjacent_label
                label -= 1
            label += 1
    
    # This part is used to visualize slic implementation. (plotting external pixels)
    points = []
    for i in range(len(centers)):
        points.append([])
        
    for i in range(row):
        for j in range(column):
            points[new_clusters[i][j]].append([i, j])
        
    slic_output = image.copy()
    is_taken = np.zeros(slic_output.shape[:2], np.bool)
    for i in range(len(points)):
        items = points[i]
        for item in items:
            x = item[0]
            y = item[1]
            cnt = 0
            if x > 0 and x < row - 1 and y > 0 and y < column - 1:
                if is_taken[x + 1, y + 1] == False and [x + 1, y + 1] not in items:
                    cnt += 1
                if is_taken[x + 1, y] == False and [x + 1, y] not in items:
                    cnt += 1
                if is_taken[x + 1, y - 1] == False and [x + 1, y - 1] not in items:
                    cnt += 1
                if is_taken[x, y + 1] == False and [x, y + 1] not in items:
                    cnt += 1
                if is_taken[x, y - 1] == False and [x, y - 1] not in items:
                    cnt += 1
                if is_taken[x - 1, y + 1] == False and [x - 1, y + 1] not in items:
                    cnt += 1
                if is_taken[x - 1, y] == False and [x - 1, y] not in items:
                    cnt += 1
                if is_taken[x - 1, y - 1] == False and [x - 1, y - 1] not in items:
                    cnt += 1
            
            if cnt >= 2:
                is_taken[x, y] == True
                slic_output[x][y] = [0, 0, 0]
            else:
                slic_output[x][y] = [slic_output[x][y][2], slic_output[x][y][1], slic_output[x][y][0]]

    cv2.imwrite(os.path.join(OUTPUT_PATH , image_name[:-4] + '-slic-2.jpg'), np.array(slic_output))

    # This part is used to visualize slic implementation. (averaging the pixels in the cluster)
    slic_output = image.copy()
    for point in points:
        if len(point) != 0:
            r = g = b = 0
            for item in point:
                rgb = image[item[0]][item[1]]
                r += rgb[0]
                g += rgb[1]
                b += rgb[2]
            r = r // len(point)
            g = g // len(point)
            b = b // len(point)
            for item in point:
                slic_output[item[0]][item[1]] = [b, g, r]
            
    cv2.imwrite(os.path.join(OUTPUT_PATH , image_name[:-4] + '-slic-1.jpg'), np.array(slic_output))
    

"""
    This function is used to save .mat files as jpg.
    @param file_name: name of the image reading from GROUND_TRUTH folder.
"""
def save_ground_truth_images(file_name):
    image_informations = loadmat(os.path.join(GROUND_TRUTH_PATH , file_name))
    # get the first ground truth image.
    ground_truth = image_informations['groundTruth'].item(0)
    # get the first segmented image.
    segmentation = ground_truth[0][0][0]
    
    # in order to display the segmented fields in the image, each field is multiplied by a certain number.
    maximum_number = max([max(row) for row in segmentation])
    for i in range(len(segmentation)):
        for j in range(len(segmentation[0])):
            segmentation[i][j] = int(segmentation[i][j] / maximum_number * 255)
    
    # extension deleted from string 'mat'
    file_name = file_name[ : -3] + 'jpg'
    write_image(file_name, GROUND_TRUTH_PATH, segmentation)


""" This is where the code starts """
if __name__ == '__main__':
    # this part is used for get segmentation of the Berkeley images
    mat_files = os.listdir(GROUND_TRUTH_PATH)
    for file_name in mat_files:
        if file_name.endswith('.mat'):
            save_ground_truth_images(file_name)

    images_list = os.listdir(FOLDER_PATH)
    for image_name in images_list:
        image = read_image(image_name)
        show_image(image_name, image)
        slic(image_name, image)     


"""
    Piksel değerlerini grayscale olarak dönüştür.
    Bir graph yapısı kur. Bu graph yapısında koordinat değerleri, piksek değerşleri ve birbirine bitişik olan
alanları birleştir.
    Birleştirme sonucunda birleşen alanların diğer alanlarla benzerliği yeni alana geçmiş olacaktır.
    Bu işlemi birleşmeyene kadar devam ettir.
    
"""