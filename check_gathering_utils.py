import os

import cv2
import numpy as np
from math import sqrt
from scipy.spatial import KDTree


def check_bboxes(bboxes, npys, threshold=200):
    """Check the Euclidean distance between the lower edge midpoints of bounding boxes.

    bboxes: list of bounding boxes, each defined as (cx, cy, w, h, conf, cls_id)
    threshold: distance threshold for warning
    """
    midpoints = [get_lower_edge_midpoint(bbox, npys) for bbox in bboxes]  # List of midpoints

    # Extract the coordinates and class ids for KDTree
    coords = [(x[0], y[0]) for x, y, cls_id in midpoints]

    # Build KDTree
    tree = KDTree(coords)

    high_risk_indices = set()

    for idx_i in range(len(coords)):
        # print(f"\nPerson {idx_i}: ({coords[idx_i]})")
        # Find points with distance within threshold radius
        indices = tree.query_ball_point(coords[idx_i], threshold)
        # print("indices:")
        # for idx_j in indices:
        #     print(f"--->Person {idx_j}: ({coords[idx_j]})")
        same_person_indices = tree.query_ball_point(coords[idx_i], 200)
        # print("same_person_indices: ")
        # for idx_j in same_person_indices:
        #     print(f"--->Person {idx_j}: ({coords[idx_j]})")
        indices = list(set(indices) - set(same_person_indices))
        # print("indices: ")
        # for idx_j in indices:
        #     print(f"--->Person {idx_j}: ({coords[idx_j]})")

        for idx_j in indices:
            if bboxes[idx_i][5] == 1 or bboxes[idx_j][5] == 1:
                high_risk_indices.update([idx_i, idx_j])
                print(f"BBoxes {idx_i} and {idx_j} are likely fighting.")
            else:
                print(f"BBoxes {idx_i} and {idx_j} are close but seems safe.")

        # # Check triplets within the neighborhood
        # for j in range(len(indices)):
        #     for k in range(j + 1, len(indices)):
        #         idx_j = indices[j]
        #         idx_k = indices[k]
        #
        #         if (idx_i, idx_j, idx_k) in processed_pairs:
        #             continue
        #
        #         if idx_i in [idx_j, idx_k]:
        #             continue
        #
        #         # Check if idx_j and idx_k are also close to each other
        #         if euclidean_distance(coords[idx_j], coords[idx_k]) < threshold:
        #             # Check if any of the midpoints belong to classes 1 or 4
        #             if any(midpoints[idx][2] in {1, 4} for idx in (idx_i, idx_j, idx_k)):
        #                 print(f"Danger: BBoxes {idx_i}, {idx_j}, and {idx_k} are closer than {threshold} pixels.")
        #             else:
        #                 print(f"Warning: BBoxes {idx_i}, {idx_j}, and {idx_k} are closer than {threshold} pixels.")
        #
        #             high_risk_indices.update([idx_i, idx_j, idx_k])
        #             processed_pairs.add((idx_i, idx_j, idx_k))
        #             processed_pairs.add((idx_j, idx_i, idx_k))
        #             processed_pairs.add((idx_k, idx_i, idx_j))

    # Change all fighting classes to stand first
    for idx in range(len(coords)):
        if bboxes[idx][5] == 1:
            bboxes[idx][5] = 4
            if idx not in high_risk_indices:
                print(f"BBoxes {idx} is likely fighting alone.")
    # Update the class id for high risk bounding boxes
    for idx in high_risk_indices:
        bboxes[idx][5] = 1

    return bboxes


def get_lower_edge_midpoint(bbox, npys):
    """Returns the lower edge midpoint of the bounding box.

    bbox: list or tuple of (cx, cy, w, h, conf, cls_id)
    """
    # cx, cy, w, h, conf, cls_id = bbox
    # lower_midpoint_u = cx
    # lower_midpoint_v = cy + h / 2
    x1, y1, x2, y2, conf, cls_id = bbox
    lower_midpoint_u = (x1 + x2) / 2
    lower_midpoint_v = max(y1, y2)
    lower_midpoint_u = lower_midpoint_u / 640 * 1920
    lower_midpoint_v = lower_midpoint_v / 480 * 1080
    lower_midpoint_X, lower_midpoint_Y, _ = image_to_world((lower_midpoint_u, lower_midpoint_v), npys)
    return (lower_midpoint_X, lower_midpoint_Y, cls_id)


def image_to_world(i, npys):
    """
    Convert image coordinate (u, v) to 2D world coordinate (X, Y) using camera calibration parameters,
    assuming Z = 0.

    :param i: Image coordinate [u,v] (2x1)
    :param M: Camera intrinsic matrix (3x3)
    :param D: Distortion coefficients (1x5 or 1x4)
    :param r_vec: Rotation vector (3x1)
    :param T: Translation vector (3x1)
    :return w: World coordinate (X, Y)
    """
    M, D, r_vec, t_vec = npys

    i_vector = np.append(i, 1).reshape(3, 1)
    M_inv = np.linalg.inv(M)
    rot_mat, _ = cv2.Rodrigues(r_vec)  # Convert rotation vector to rotation matrix
    rot_mat_inv = np.linalg.inv(rot_mat)
    left_side = np.matmul(rot_mat_inv, np.matmul(M_inv, i_vector))  # (3x1)
    right_side = np.matmul(rot_mat_inv, t_vec)  # (3x1)
    s = (0 + right_side[2, 0]) / left_side[
        2, 0]  # 0 mean the Z of the world coordinate locate in the same plane with the orgin
    # w = (s * left_side - right_side) # unit is how many box(need to multiply by the length of the box)
    w = np.matmul(rot_mat_inv, (s * np.matmul(M_inv,
                                              i_vector)) - t_vec)  # unit is how many box(need to multiply by the length of the box)
    return w


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points.

    point1, point2: tuples of (x, y)
    """
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def draw_axes(img, npys, length=130):
    """
    Draw XYZ axes on the image.

    :param img: Input image
    :param M: Camera intrinsic matrix (3x3)
    :param r_vec: Rotation vector (3x1)
    :param T: Translation vector (3x1)
    :param length: Length of the axes (default: 1( 2.5cm ))
    """

    M, D, r_vec, T = npys

    # Define the axes' endpoints in world coordinates
    origin = np.array([0, 0, 0, 1]).reshape(4, 1)
    x_axis = np.array([length, 0, 0, 1]).reshape(4, 1)
    y_axis = np.array([0, length, 0, 1]).reshape(4, 1)
    z_axis = np.array([0, 0, length, 1]).reshape(4, 1)

    # Transform world coordinates to image coordinates
    rot_mat, _ = cv2.Rodrigues(r_vec)
    transform = np.column_stack((rot_mat, T))
    img_origin = np.dot(M, np.dot(transform, origin))
    img_x_axis = np.dot(M, np.dot(transform, x_axis))
    img_y_axis = np.dot(M, np.dot(transform, y_axis))
    img_z_axis = np.dot(M, np.dot(transform, z_axis))

    # Normalize the coordinates
    img_origin = (img_origin / img_origin[2]).astype(int)[:2].ravel()
    img_x_axis = (img_x_axis / img_x_axis[2]).astype(int)[:2].ravel()
    img_y_axis = (img_y_axis / img_y_axis[2]).astype(int)[:2].ravel()
    img_z_axis = (img_z_axis / img_z_axis[2]).astype(int)[:2].ravel()

    for coords in [img_origin, img_x_axis, img_y_axis, img_z_axis]:
        coords[0] = coords[0] / 1920 * 640
        coords[1] = coords[1] / 1080 * 480


    # Draw the axes
    img = cv2.line(img, tuple(img_origin), tuple(img_z_axis), (0, 0, 255), 2)  # Z - blue
    cv2.putText(img, 'Z', tuple(img_z_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    img = cv2.line(img, tuple(img_origin), tuple(img_x_axis), (255, 0, 0), 2)  # X - red
    cv2.putText(img, 'X', tuple(img_x_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    img = cv2.line(img, tuple(img_origin), tuple(img_y_axis), (0, 255, 0), 2)  # Y - green
    cv2.putText(img, 'Y', tuple(img_y_axis), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img
