
import shutil

import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
from scipy.ndimage.filters import convolve
from scipy import ndimage
import utils

M_MATRIX_SIZE = 2
K = 0.04
COLS = 0
ROWS = 1
S = 1
TOP_MATCHES = -2
LAST_ROW = -1
EUCLIDEAN_NORM = 2
X = 0
Y = 1
HOMOGRAPHY_WIDTH = 3
HOMOGRAPHY_LENGTH = 3
LAST = -1


def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    x_conv_matrix = np.array([[1, 0, -1]])
    R = create_response_image(im, x_conv_matrix)
    local_maxima = non_maximum_suppression(R)
    return create_harris_feature_points(local_maxima)


def create_response_image(im, convolution_matrix):
    """
    return R matrix, which is det_M - K * (trace**2) as explained in the pdf
    """
    I_X = convolve(im, convolution_matrix)
    I_Y = convolve(im, convolution_matrix.T)
    I_X_blurred = utils.blur_spatial(I_X ** 2, 3)
    I_Y_blurred = utils.blur_spatial(I_Y ** 2, 3)
    I_X_I_Y_blurred = utils.blur_spatial(I_X * I_Y, 3)
    M = np.stack((I_X_blurred, I_X_I_Y_blurred, I_X_I_Y_blurred, I_Y_blurred), axis=2). \
        reshape((im.shape[0], im.shape[1], M_MATRIX_SIZE, M_MATRIX_SIZE))
    det_M = np.linalg.det(M)
    trace = np.trace(M, axis1=2, axis2=3)
    R = det_M - K * (trace ** 2)
    return R


def create_harris_feature_points(boolean_matrix):
    """
    create matrix where only the harris points are non-zero values.
    """
    x_points, y_points = np.where(boolean_matrix)
    num_of_coordinates = len(y_points)
    xy_coordinates = np.zeros([num_of_coordinates, 2]).astype(np.int64)
    xy_coordinates[:, ROWS] = x_points
    xy_coordinates[:, COLS] = y_points
    return xy_coordinates


def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    K = 1 + 2*desc_rad
    N = pos.shape[0]
    descriptor_3D_array = np.zeros([N, K, K])

    i_th_descriptor = 0
    for corner_point in pos:
        # find interpolated patch:
        x_loc, y_loc = corner_point
        axis_x_start_end = np.arange(y_loc - desc_rad, y_loc + desc_rad + S)
        axis_y_start_end = np.arange(x_loc - desc_rad, x_loc + desc_rad + S)
        patch =  np.meshgrid(axis_x_start_end, axis_y_start_end)
        interpolated_patch = ndimage.map_coordinates(im, patch, order=1, prefilter=False)
        # find the normalized descriptor and insert in descriptor_3D_array:
        denominator = np.linalg.norm(interpolated_patch - np.mean(interpolated_patch)) #denomerator = MEHANE in hebrew
        patch_mean = np.mean(interpolated_patch)
        patch = ((interpolated_patch - patch_mean) / denominator) if denominator != 0 else np.zeros([7, 7])
        descriptor_3D_array[i_th_descriptor, :, :] = patch
        i_th_descriptor += 1

    return descriptor_3D_array


def find_features(pyr):
  """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image. 
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
  corners = spread_out_corners(pyr[0], 7, 7, 3)
  descriptors = sample_descriptor(pyr[2], corners/4, 3)
  return [corners, descriptors]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
    """
    desc1_rows, desc1_patch = desc1.shape[0], desc1.shape[1]
    desc2_rows, desc2_patch = desc2.shape[0], desc2.shape[1]

    desc1_flattened, desc2_flattened = compute_flattened_descriptor_matrixes(desc1, desc2, desc2_patch, desc2_rows)

    desc1_match_points, desc2_match_points = create_match_points(desc1_flattened, desc1_rows, desc2_flattened,
                                                                 desc2_rows, min_score)
    desc1_match_points = desc1_match_points.astype(np.int)
    desc2_match_points = desc2_match_points.astype(np.int)
    return desc1_match_points, desc2_match_points


def create_match_points(desc1_flattened, desc1_rows, desc2_flattened, desc2_rows, min_score):
    """
    :return: return only the points that answer the 3 demands, as presented in the pdf
    """
    scores = np.dot(desc1_flattened, desc2_flattened).reshape(desc1_rows, desc2_rows)
    top2row_position = np.partition(scores, TOP_MATCHES, axis=ROWS)[:, TOP_MATCHES:][:, 0].reshape(desc1_rows, 1)
    top2COL_position = np.partition(scores, TOP_MATCHES, axis=COLS)[TOP_MATCHES:, ][0, :].reshape(1, desc2_rows)
    desc1_match_points, desc2_match_points = np.where((scores >= min_score) &
                                                      (scores >= top2row_position) &
                                                      (scores >= top2COL_position))
    return desc1_match_points, desc2_match_points


def compute_flattened_descriptor_matrixes(desc1_matrix, desc2_matrix, desc2_patch, desc2_rows):
    """
    returned the descriptor matrixes in the correct shape
    """
    desc2_patch_size = desc2_patch * desc2_patch
    desc1_flattened = desc1_matrix.reshape(desc1_matrix.shape[0], -1)
    desc2_flattened = desc2_matrix.reshape([desc2_rows, desc2_patch_size, 1])
    return desc1_flattened, desc2_flattened


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    N = pos1.shape[0]
    ones_matrix = np.ones((N,1))
    pos1_homogen = np.hstack([pos1, ones_matrix])
    compute_homography = np.dot(pos1_homogen, H12.T)
    z_row = compute_homography[:, LAST_ROW]
    z_row = np.reshape(z_row, (N,1))
    x_y_rows = compute_homography[:,:2]
    return x_y_rows / z_row


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    best_inliers_amount = 0
    best_inlier_indexes = None
    N = points2.shape[0]

    for i in range(num_iter):
      indexes = np.random.permutation(N)[0:2]
      H12 = estimate_rigid_transform(points1[indexes], points2[indexes], translation_only)
      point2_tag = apply_homography(points1, H12)

      Ej = np.linalg.norm(point2_tag - points2, EUCLIDEAN_NORM, axis=1)
      current_inliers = Ej < inlier_tol
      current_sum = np.sum(current_inliers)
      if best_inliers_amount < current_sum:
          best_inliers_amount = current_sum
          best_inlier_indexes = current_inliers

    point1_loc = points1[best_inlier_indexes]
    points2_loc = points2[best_inlier_indexes]
    H = estimate_rigid_transform(point1_loc, points2_loc, translation_only)
    return H, best_inlier_indexes


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispaly matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An array shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    # create inliers
    points2_new = points2.copy()
    shift_amount= im1.shape[1]
    points2_new[:,0] += shift_amount
    im1_inliers = points1[inliers]
    im2_inliers = points2_new[inliers]

    # create outliers
    N = points1.shape[0]
    outliers = np.ones(N, np.bool)
    outliers[inliers] = False
    im1_outliers = points1[outliers]
    im2_outliers = points2_new[outliers]

    # plot
    both_images = np.hstack((im1, im2))
    plt.imshow(both_images, cmap='gray')
    plt.plot([im1_outliers[:, X], im2_outliers[:, X]], [im1_outliers[:, Y], im2_outliers[:, Y]],
             mfc='r', c='b', lw=.4, ms=4, marker='o')
    plt.plot([im1_inliers[:, X], im2_inliers[:, X]], [im1_inliers[:, Y], im2_inliers[:, Y]],
             mfc='r', c='y', lw=.4, ms=4, marker='o')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
    Convert a list of successive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
    accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
    where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    M = len(H_succesive)

    homographies_lst = np.zeros((M + 1, HOMOGRAPHY_WIDTH, HOMOGRAPHY_LENGTH))
    reference_point = np.eye(3)
    homographies_lst[m] = reference_point
    compute_homographies_before_m(H_succesive, homographies_lst, m)
    compute_homographies_after_m(H_succesive, homographies_lst, m)
    homographies_lst = list(homographies_lst)
    return homographies_lst


def compute_homographies_before_m(H_succesive, homographies_lst, m):
    """
    create homographies in all places PRIOR to location m in the homography array
    """
    for i in range(m - 1, -1, -1):
        current_H = H_succesive[i]
        current_homography = homographies_lst[i + 1]
        homographies_lst[i] = np.dot(current_homography, current_H)
        homographies_lst[i] /= homographies_lst[i, 2, 2]


def compute_homographies_after_m(H_succesive, homographies_lst, m):
    """
    create homographies in all places AFETR to location m in the homography array
    """
    for j in range(m, len(H_succesive)):
        current_H = H_succesive[j]
        current_homography = homographies_lst[j]
        homographies_lst[j + 1] = np.dot(current_homography, np.linalg.inv(current_H))
        homographies_lst[j + 1] /= homographies_lst[j + 1, 2, 2]


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
    and the second row is the [x,y] of the bottom right corner
    """
    corner_arr = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    corner_homography = apply_homography(corner_arr, homography)

    top_left = find_top_left(corner_homography)
    bottom_right = find_bottom_right(corner_homography)

    final_arr = np.array([top_left, bottom_right], dtype=np.int)
    return final_arr


def find_bottom_right(corner_homography):
    """
    find largest X and largest Y (top right corner is 0,0)
    :return: bottom right point of the new image
    """
    bottom_right_X = np.ceil(np.max(corner_homography[:, 0]))
    bottom_right_Y = np.ceil(max(corner_homography[:, 1]))
    bottom_right = [bottom_right_X, bottom_right_Y]
    return bottom_right


def find_top_left(corner_homography):
    """
    find smallest X and smallest Y (top right corner is 0,0)
    :return: bottom right point of the new image
    """
    top_left_X = np.floor(np.min(corner_homography[:, 0]))
    top_left_Y = np.floor(min(corner_homography[:, 1]))
    top_left = [top_left_X, top_left_Y]
    return top_left


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """
    H_inv = np.linalg.inv(homography)
    H_inv /= H_inv[2,2]

    image_width, image_height = image.shape[Y], image.shape[X]
    corners = compute_bounding_box(homography, image_width, image_height)
    warped_image_height = corners[1, 1] - corners[0, 1] + 1
    warped_image_width = corners[1, 0] - corners[0, 0] + 1

    meshed_coordinates = get_meshed_coordinates(corners)
    coord_img = np.asarray([meshed_coordinates[0].reshape(LAST), meshed_coordinates[1].reshape(LAST)]).T
    after_homography_apply = apply_homography(coord_img, H_inv)

    final_image_coordinates = (after_homography_apply[:, ROWS], after_homography_apply[:,COLS])
    final_image = ndimage.map_coordinates(image, final_image_coordinates, order=1, prefilter=False)
    final_image = final_image.reshape(warped_image_height, warped_image_width)
    return final_image


def get_meshed_coordinates(corners):
    x_values = np.arange(corners[0, 0], corners[1, 0] + 1)
    y_values = np.arange(corners[0, 1], corners[1, 1] + 1)
    meshed_coordinates = np.meshgrid(x_values, y_values)
    return meshed_coordinates


def warp_image(image, homography):
  """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
  return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0,-1]
  for i in range(1, len(homographies)):
    if homographies[i][0,-1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0,-1]
  return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2,:2] = rotation
  H[:2, 2] = translation
  return H


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret


def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0,2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) & 
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  ret = corners[legit,:]
  return ret


class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in self.files:
      image = utils.read_image(file, 1)
      self.h, self.w = image.shape
      pyramid, _ = utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]


  def generate_panoramic_images(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]


  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
        plt.imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))


  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()



