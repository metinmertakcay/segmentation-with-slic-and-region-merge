# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 22:27:16 2019
@author: Metin Mert Akçay
"""
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.io import loadmat
import numpy as np
import cv2
import sys
import os


MIN_SUPERPIXEL_AREA = 10
MAXIMUM_ITERATION = 10
FOLDER_PATH = 'Images'
OUTPUT_PATH = 'Output'
GROUND_TRUTH_PATH = 'Ground_truth'
K = 9000
M = 10


class file_content():
    def __init__(self):
        self.image_dict = {}
    
    def is_image_name_checked(self, image_name):
        if image_name in self.image_dict:
            return True
        return False
    
    def append_variable(self, image_name, score):
        self.image_dict[image_name] = score
        
    def get_image_score(self, image_name):
        return self.image_dict.get(image_name)


class graph():
    def __init__(self, id):
        self.id = id
        self.pixel_list = []
        self.grayscale_pixel_list = []
        self.neighbour_list = []
        self.coordinate_list = []
        self.histogram = []
        self.combined = False   
    
    def get_id(self):
        return self.id
    
    def get_pixel_value(self):
        return self.pixel_list
    
    def append_pixel_value(self, pixel):
        self.pixel_list.append(pixel)
    
    def extend_pixel_value(self, pixel_list):
        self.pixel_list.extend(pixel_list)
        
    def get_grayscale_pixel_value(self):
        return self.grayscale_pixel_list
    
    def append_grayscale_pixel_value(self, pixel):
        self.grayscale_pixel_list.append(pixel)
    
    def extend_grayscale_pixel_value(self, pixel_list):
        self.grayscale_pixel_list.extend(pixel_list)

    def mean_grayscale_pixel_value(self):
        return sum(self.grayscale_pixel_list) / len(self.grayscale_pixel_list)

    def get_neighbour(self):
        return self.neighbour_list

    def append_neighbour(self, id):
        if id not in self.neighbour_list:
            self.neighbour_list.append(id)

    def extend_neighbour(self, id_list):
        for id in id_list:
            if id not in self.neighbour_list and id != self.id:
                self.neighbour_list.append(id)

    def get_coordinate(self):
        return self.coordinate_list

    def append_coordinate(self, coordinate):
        self.coordinate_list.append(coordinate)
        
    def extend_coordinate(self, coordinate_list):
        self.coordinate_list.extend(coordinate_list)

    def get_histogram(self):
        return self.histogram
    
    def set_histogram(self, histogram):
        self.histogram = histogram
        
    def is_combined(self):
        return self.combined
    
    def set_combined(self):
        self.combined = True

""" 
    This function is used to read image.
    @param image_name: name of the image.
    @return image: image.
"""
def read_image(image_name):
    image = cv2.imread(os.path.join(FOLDER_PATH, image_name))
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b]) # b,g,r image is converted to r,g,b.
    return image


"""
    This function is used for display image on the console.
    @param image_name: name of the image.
    @param image: image.
"""
def show_image(image_name, image):
    plt.imshow(np.array(image))
    plt.title(image_name)
    plt.show()    


""" 
    This function is used to write the image to the file.
    @param image_name: name of the image to be saved
    @param folder_name: folder name to be saved
    @param image: image.
"""
def write_image(image_name, folder_path, image):
    if(not(os.path.exists(folder_path))):
        os.makedirs(folder_path)
    cv2.imwrite(os.path.join(folder_path , image_name), np.array(image))


"""
    This function is used to find distance between two point.
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
    This function is used to convert r, g, b to grayscale
    @param r, g, b: pixel values
    @return grayscale value
"""
def rgb_to_grayscale(r, g, b):
    return (0.3 * r) + (0.59 * g) + (0.11 * b);


"""
    This function is used to find local minimum gradient. (3 x 3)
    @param center: Mid point for S x S area
    @param image: image
"""
def local_minimum_gradient(center, image):
    min_gradient = sys.maxsize
    local_min_point = center
    
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            r1, g1, b1 = image[i + 1][j]
            r2, g2, b2 = image[i][j + 1]
            r3, g3, b3 = image[i][j]
            p1 = rgb_to_grayscale(r1, g1, b1)
            p2 = rgb_to_grayscale(r2, g2, b2)
            p3 = rgb_to_grayscale(r3, g3, b3)
            if ((p1 - p3) ** 2) ** 0.5 + ((p2 - p3) ** 2) ** 0.5 < min_gradient:
                min_gradient = ((p1 - p3) ** 2) ** 0.5 + ((p2 - p3) ** 2) ** 0.5
                local_min_point = [i, j]
    return local_min_point


"""
    This function is used to create a list which holds point coordinates. (x, y coordinate)
    @param row - column: These values ​​are used to determine the size of the list.
"""
def create_cluster_list(row, column):
    clusters_row = []
    for i in range(row):
        clusters_column = []
        for j in range(column):
            clusters_column.append(0)
        clusters_row.append(clusters_column)
    return clusters_row


"""
    This function is used to create a list which keep distances of two point.
    @param row - column: These values ​​are used to determine the size of the list.
"""
def create_distance_list(row, column):
    distances_row = []
    for i in range(row):
        distances_column = []
        for j in range(column):
            distances_column.append(sys.maxsize)
        distances_row.append(distances_column)
    return distances_row


"""
    This function is used to create a list which count points in the clusters.
    @param size: size of list
"""
def create_count_list(size):
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
    This function is used to visualize the slic algorithm output.
    @param points: points in the clusters.
    @param image: image.
    @param image_name: name of the image.
    @param operation: used to determine which action was taken.
"""
def show_output_of_slic(points, image, image_name, operation):
    output = image.copy()
    row, column = image.shape[:2]
    is_taken = np.zeros(output.shape[:2], np.bool)
    for i in range(len(points)):
        items = points[i]
        if len(items) > MIN_SUPERPIXEL_AREA:
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
                    output[x][y] = [0, 0, 0]
                else:
                    output[x][y] = [output[x][y][2], output[x][y][1], output[x][y][0]]
        else:
            for item in items:
                x = item[0]
                y = item[1]
                output[x][y] = [output[x][y][2], output[x][y][1], output[x][y][0]]
    cv2.imwrite(os.path.join(OUTPUT_PATH, image_name[:-4] + operation), np.array(output))


"""
    This function is used to visualize slic algorithm output. (averaging the pixels in the cluster)
    @param points: it is a list which store location of the pixels.
    @param image: image read from file.
    @param image_name: name of the image read from the file.
    @param operation: used to determine which action was taken.
"""
def show_slic_output_taking_average_of_pixels_in_cluster(points, image, image_name, operation):
    output = image.copy()
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
                output[item[0]][item[1]] = [b, g, r]        
    cv2.imwrite(os.path.join(OUTPUT_PATH , image_name[:-4] + operation), np.array(output))


"""
    This function is used to visualize slic and region merge operation output. (averaging the pixels in the cluster)
    @param superpixel_ids: non - merged super pixel ids as a result of region merge.
    @param graph_list: it is a list which contains graph objects like neighbours, pixel values, pixel location etc.
    @param image: image.
    @param image_name: name of the image.
    @param operation: used to determine which action was taken.
"""
def show_output_taking_average_of_pixels_in_cluster(superpixel_ids, graph_list, image, image_name, operation):
    output = image.copy()
    for i in superpixel_ids:
        bgr = [0, 0, 0]
        # location information of the pixels in the cluster is received.
        for coordinate in graph_list[i].get_coordinate():
            x = coordinate[0]
            y = coordinate[1]
            bgr[0] += image[x][y][0]
            bgr[1] += image[x][y][1]
            bgr[2] += image[x][y][2]
        number_of_point = len(graph_list[i].get_coordinate())
        bgr = [value // number_of_point for value in bgr]
        # assigning new pixel values.
        for coordinate in graph_list[i].get_coordinate():
            x = coordinate[0]
            y = coordinate[1]
            output[x][y] = [bgr[2], bgr[1], bgr[0]]
    cv2.imwrite(os.path.join(OUTPUT_PATH, image_name[:-4] + operation), np.array(output))
    

"""
    Slic algorithm is implemented in this function. Over-segmentation was performed with using slic.
    @param image_name: name of the image
    @param image: image
"""
def slic(image_name, image):
    row, column = image.shape[:2]
    
    N = row * column            # number of pixel in image
    S = int((N / K) ** 0.5)     # Approximately a super-pixel area (S x S)
    
    # initialize cluster centers
    centers = []
    for i in range(int(S / 2), row - int(S / 2), S):
        for j in range(int(S / 2), column - int(S / 2), S):
            # find local minimum gradient
            local_point = local_minimum_gradient([i, j], image)
            r, g, b = image[local_point[0]][local_point[1]]
            center = [r, g, b, local_point[0],local_point[1]]
            centers.append(center)
    
    # clusters: it is a list which keeps closest points.
    # distance: it is a list which keeps shortest distance between point and centers.
    # number_of_points_in_cluster: it is a list which holds the number of points in the cluster.
    clusters = create_cluster_list(row, column)
    distances = create_distance_list(row, column)
    number_of_points_in_cluster = create_count_list(len(centers))
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
            centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0
            number_of_points_in_cluster[j] = 0

        # calculate new x and y values of center point and the number of points in cluster
        for j in range(row):
            for k in range(column):
                center_id = clusters[j][k]
                centers[center_id][0] += image[j][k][0]
                centers[center_id][1] += image[j][k][1]
                centers[center_id][2] += image[j][k][2]
                centers[center_id][3] += j
                centers[center_id][4] += k
                number_of_points_in_cluster[center_id] += 1

        # assign new centers r,g,b values        
        for j in range(len(centers)):
            if number_of_points_in_cluster[j] != 0:
                centers[j][0] = centers[j][0] // number_of_points_in_cluster[j]
                centers[j][1] = centers[j][1] // number_of_points_in_cluster[j]
                centers[j][2] = centers[j][2] // number_of_points_in_cluster[j]
                centers[j][3] = centers[j][3] // number_of_points_in_cluster[j]
                centers[j][4] = centers[j][4] // number_of_points_in_cluster[j]
    
    
    # this part is used to enforce connectivity
    label = 0
    adjacent_label = 0
    limit = (row * column) // (len(centers) * 4) # size of the superpixel
    new_clusters = -1 * np.ones(image.shape[:2]).astype(np.int64)
    
    items = []
    dx_4 = [-1, 0, 1, 0]
    dy_4 = [0, -1, 0, 1]
    for i in range(row):
        for j in range(column):
            if new_clusters[i][j] == -1:
                items = []
                items.append((i, j))
                for dx, dy in zip(dx_4, dy_4):
                    x = items[0][0] + dx
                    y = items[0][1] + dy
                    if (x >= 0 and x < row and y >= 0 and y < column and new_clusters[x][y] >= 0):
                        adjacent_label = new_clusters[x][y]
            
            k = 0
            item_count = 1
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
            if item_count <= limit:
                for k in range(item_count):
                    x = items[k][0]
                    y = items[k][1]
                    new_clusters[x][y] = adjacent_label
                label -= 1
            label += 1
    
    total_cluster = max([max(row) for row in new_clusters]) + 1
    # This part is used to visualize slic implementation. (plotting external pixels)
    points = []
    for i in range(total_cluster):
        points.append([])
        
    for i in range(row):
        for j in range(column):
            points[new_clusters[i][j]].append([i, j])
    
    # visualize algorithm output
    show_slic_output_taking_average_of_pixels_in_cluster(points, image, image_name, '-slic-1.jpg')
    show_output_of_slic(points, image, image_name, '-slic-2.jpg')
    
    return new_clusters


"""
    This function is used to determine the neighborhoods of clusters.
    @param image: image.
    @param image_name: name of the image.
    @param clusters: it is a variable which holds cluster information (point - id)
"""
def find_neighbour(image, image_name, clusters):
    row, column = image.shape[:2]    
    total_cluster = max([max(row) for row in clusters]) + 1
    
    graph_list = []
    for i in range(total_cluster):
        graph_list.append(graph(i))

    for i in range(row):
        for j in range(column):
            node = graph_list[clusters[i][j]]
            
            r, g, b = image[i][j]
            node.append_coordinate([i, j])
            node.append_pixel_value([r, g, b])
            node.append_grayscale_pixel_value(rgb_to_grayscale(r, g, b))
            if i + 1 < row and j + 1 < column and clusters[i + 1][j + 1] != clusters[i][j]:
                node.append_neighbour(clusters[i + 1][j + 1])
            if i + 1 < row and clusters[i + 1][j] != clusters[i][j]:
                node.append_neighbour(clusters[i + 1][j])
            if i + 1 < row and j - 1 > 0 and clusters[i + 1][j - 1] != clusters[i][j]:
                node.append_neighbour(clusters[i + 1][j - 1])
            if j + 1 < column and clusters[i][j + 1] != clusters[i][j]:
                node.append_neighbour(clusters[i][j + 1])
            if j - 1 > 0 and clusters[i][j - 1] != clusters[i][j]:
                node.append_neighbour(clusters[i][j - 1])
            if i - 1 > 0 and j + 1 < column and clusters[i - 1][j + 1] != clusters[i][j]:
                node.append_neighbour(clusters[i - 1][j + 1])
            if i - 1 > 0 and clusters[i - 1][j] != clusters[i][j]:
                node.append_neighbour(clusters[i - 1][j])
            if i - 1 > 0 and j - 1 > 0 and clusters[i - 1][j - 1] != clusters[i][j]:
                node.append_neighbour(clusters[i - 1][j - 1])          
    
    for i in range(total_cluster):
        histogram = cv2.calcHist([np.array([graph_list[i].get_pixel_value()])], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
        graph_list[i].set_histogram(histogram)
    
    thresh1 = 0.5
    thresh2 = 20
    superpixel_id_list = []
    for i in range(total_cluster):
        if graph_list[i].is_combined() == False:    
            superpixel_id_list.append(i)
            region_merge(i, graph_list, thresh1, thresh2)

    show_output_taking_average_of_pixels_in_cluster(superpixel_id_list, graph_list, image, image_name, '-output-1.jpg')
    
    image = np.ones((image.shape[0], image.shape[1]), dtype = int)
    for i in range(len(superpixel_id_list)):
        superpixel_id = superpixel_id_list[i]
        coordinates = graph_list[superpixel_id].get_coordinate()
        for coordinate in coordinates:
            x = coordinate[0]
            y = coordinate[1]
            image[x][y] = superpixel_id
    return image.flatten().tolist()
    

"""
    This function merges areas that have been over segmented. (histogram differences + pixel value differences)
    @param i: super pixel id value.
    @param graph_list: It includes neighbours, pixel values, pixel location etc.
    @param tresh1: histogram differences threshold value.
    @param tresh2: pixel value differences threshold value.
"""
def region_merge(i, graph_list, thresh1, thresh2):
    for neighbour in graph_list[i].get_neighbour():
        if graph_list[neighbour].is_combined() == False:
            if(cv2.compareHist(graph_list[i].get_histogram(), graph_list[neighbour].get_histogram(), cv2.HISTCMP_INTERSECT) > thresh1 or
               np.absolute(graph_list[i].mean_grayscale_pixel_value() - graph_list[neighbour].mean_grayscale_pixel_value()) < thresh2):
                graph_list[neighbour].set_combined()
                graph_list[i].extend_pixel_value(graph_list[neighbour].get_pixel_value())
                graph_list[i].extend_grayscale_pixel_value(graph_list[neighbour].get_grayscale_pixel_value())
                graph_list[i].extend_coordinate(graph_list[neighbour].get_coordinate())
                graph_list[i].extend_neighbour(graph_list[neighbour].get_neighbour())

"""
    This function is used to convert .mat to .jpg.
    @param file_name: name of the image reading from GROUND_TRUTH folder.
"""
def convert_ground_truth_extension(file_name):
    segmentation_and_boundaries_infos = loadmat(os.path.join(GROUND_TRUTH_PATH , file_name))
    
    # get images segmented by human.
    number_of_ground_truth = len(segmentation_and_boundaries_infos)
    for i in range(number_of_ground_truth):
        ground_truth = segmentation_and_boundaries_infos['groundTruth'].item(i)
        segmented_image = ground_truth[0][0][0]
        
        copy_segmented_image = segmented_image.copy()
        # in order to display the segmented fields, each field is multiplied by a certain number.
        maximum_number = max([max(row) for row in copy_segmented_image])
        for j in range(len(copy_segmented_image)):
            for k in range(len(copy_segmented_image[0])):
                copy_segmented_image[j][k] = int(copy_segmented_image[j][k] / maximum_number * 255)
        
        # extension deleted from string 'mat'
        f_name = file_name[ : -4] + ' - grayscale - ' + str(i) + '.jpg'
        write_image(f_name, GROUND_TRUTH_PATH, copy_segmented_image)
        
        # convert it colorful image
        image = read_image(file_name[ : -4] + '.jpg')

        dictionary = {}
        for j in range(1, maximum_number + 1):
            dictionary[j] = [0, 0, 0, 0]  #r, g, b and total count

        for j in range(len(segmented_image)):
            for k in range(len(segmented_image[0])):
                rgb = image[j][k]
                dictionary[segmented_image[j][k]] = np.add(dictionary.get(segmented_image[j][k]), [rgb[0], rgb[1], rgb[2], 1])
        
        for j in range(1, maximum_number + 1):
            list_inside_dict = dictionary.get(j)
            r = list_inside_dict[0] // list_inside_dict[3]
            g = list_inside_dict[1] // list_inside_dict[3]
            b = list_inside_dict[2] // list_inside_dict[3]
            dictionary[j] = [b, g, r]

        for j in range(len(segmented_image)):
            for k in range(len(segmented_image[0])):
                image[j][k] = dictionary[segmented_image[j][k]]
        
        # extension deleted from string 'mat'
        f_name = file_name[ : -4] + ' - colorful - ' + str(i) + '.jpg'
        write_image(f_name, GROUND_TRUTH_PATH, image)    


"""
    This function is used to applyied probabilistic rand index evaluation metric.
    @param image_name: image name
    @param prediction: slic and region merge algoritms result
    @param score / number_of_ground_truth: PRI result for related image
"""
def find_probabilistic_rand_index(image_name, prediction):
    segmentation_and_boundaries_infos = loadmat(os.path.join(GROUND_TRUTH_PATH , image_name[ : -4] + '.mat'))

    score = 0
    number_of_ground_truth = len(segmentation_and_boundaries_infos)
    for i in range(number_of_ground_truth):
        ground_truth = segmentation_and_boundaries_infos['groundTruth'].item(i)
        segment = ground_truth[0][0][0].flatten().tolist()
        score += rand_index_score(segment, prediction)
    print(score / number_of_ground_truth)
    return score / number_of_ground_truth            


# np.bincount: this function is used to number of passing numbers was found.
# comb: combination example (6 2) = 15, (10, 2) = 45
# np_c: concanatenation operation.
# tp: every time a pair of elements is grouped together by the two cluster
# tn: every time a pair of elements is not grouped together by the two cluster
"""
    This function is used for calculate rand index (RI) score
    @param labels_ground_truth: actual label values
    @param labels_prediction: predicted label values
"""
def rand_index_score(labels_ground_truth, labels_prediction):
    # tp = true positive, tn: true negative, fp: false positive, fn: false negative
    sum_tp_fp = comb(np.bincount(labels_ground_truth), 2).sum()
    sum_tp_fn = comb(np.bincount(labels_prediction), 2).sum()
    A = np.c_[(labels_ground_truth, labels_prediction)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(labels_ground_truth))
    fp = sum_tp_fp - tp
    fn = sum_tp_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


""" This is where the code starts """
if __name__ == '__main__':
    # This line is used to get .mat files name
    """mat_files = os.listdir(GROUND_TRUTH_PATH)
    for file_name in mat_files:
        if file_name.endswith('.mat'):
            convert_ground_truth_extension(file_name)"""
    
    if not(os.path.isfile("test_result.txt")):
        file = open("test_result.txt", "a+")
        file.close()
    
    file = open("test_result.txt", "r")
    content = file_content()
    lines = file.readlines()
    for line in lines:
        res = line.rstrip().split()
        content.append_variable(res[0], res[1])
    file.close()
    
    total_score = 0
    file = open("test_result.txt", "a+")
    images_list = os.listdir(FOLDER_PATH)
    for image_name in images_list:
        if content.is_image_name_checked(image_name):
            total_score += float(content.get_image_score(image_name))
        else:
            image = read_image(image_name)
            show_image(image_name, image)
            clusters = slic(image_name, image)     
            prediction = find_neighbour(image, image_name, clusters)
            score = find_probabilistic_rand_index(image_name, prediction)
            total_score += score
            file.write(image_name + " " + str(score) + "\n")
            file.flush()
        
    print("PRI Score: ", str(total_score / len(images_list)))
    file.close()