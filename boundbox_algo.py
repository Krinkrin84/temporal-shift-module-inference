import cv2

# ----------------------------- Define Colors for Classes ----------------------------- #
COLORS = {
    'adult': (0, 255, 0),  # Green
    'child': (0, 0, 255),  # Red
}


def resize_bbox(bbox, resize_factor, image_width, image_height):
    """
    Resizes a bounding box by a given factor around its center, checks the width to height ratio,
    crops the upper part if the ratio is smaller than 1:2, adjusts the bounding box to be square,
    and clamps it within image boundaries.

    Args:
        bbox (list or tuple): The bounding box in [x1, y1, x2, y2] format.
        resize_factor (float): The factor by which to resize the bounding box.
        image_width (int): Width of the image.
        image_height (int): Height of the image.

    Returns:
        list: The resized (and possibly squared) and clamped bounding box in [x1, y1, x2, y2] format.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    # 检查宽高比是否小于 1:2
    if width * 2 < height:
        # 计算需要裁剪的高度
        crop_height = height / 2
        # 裁剪上半部分
        y1 += crop_height
        height = height - crop_height
        y2 = y1 + height
        center_y = y1 + height / 2

    # 根据 resize_factor 调整宽度和高度
    new_width = width * resize_factor
    new_height = height * resize_factor

    # 使边界框成为正方形，取较大的边作为新边长
    max_side = max(new_width, new_height)
    new_width = new_height = max_side

    # 计算新的坐标
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    # 将坐标限制在图像边界内
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    # 重新计算宽度和高度，以确保正方形
    clamped_width = new_x2 - new_x1
    clamped_height = new_y2 - new_y1
    max_side_clamped = min(clamped_width, clamped_height)

    # 根据新的中心点调整为正方形
    new_x1 = center_x - max_side_clamped / 2
    new_y1 = center_y - max_side_clamped / 2
    new_x2 = center_x + max_side_clamped / 2
    new_y2 = center_y + max_side_clamped / 2

    # 最终限制坐标，以防止重新调整导致溢出
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]


def crop_upper_bbox(bbox, image_width, image_height, ratio_threshold=2.0):
    """
    Crops the upper part of a bounding box if its height is greater than or equal to
    a specified multiple of its width, and clamps it within image boundaries.

    Args:
        bbox (list or tuple): The bounding box in [x1, y1, x2, y2] format.
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        ratio_threshold (float): The height-to-width ratio threshold for cropping.

    Returns:
        list: The cropped (upper part only) and clamped bounding box in [x1, y1, x2, y2] format.
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # Check if the height is greater than or equal to the threshold * width
    if height >= ratio_threshold * width:
        # Crop the upper part by reducing the height
        new_height = width * ratio_threshold  # Upper part height is ratio_threshold times the width
        y2 = y1 + new_height  # Adjust the bottom boundary to crop the upper part

    # Ensure the bounding box is still within image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    return [int(x1), int(y1), int(x2), int(y2)]


def combine_overlapping_boxes(ents):
    combined_boxes = []
    seen_indices = set()

    # Loop through each box in the list
    for i, box1 in enumerate(ents):
        if i in seen_indices:
            continue

        new_box = box1.copy()  # Create a copy of the current box

        # Check for overlaps with boxes of different classes
        for j, box2 in enumerate(ents):
            if i != j and box1['class'] != box2['class']:
                # Calculate overlap between the two boxes
                x_overlap = max(0, min(box1['bbox'][2], box2['bbox'][2]) - max(box1['bbox'][0], box2['bbox'][0]))
                y_overlap = max(0, min(box1['bbox'][3], box2['bbox'][3]) - max(box1['bbox'][1], box2['bbox'][1]))
                overlap_area = x_overlap * y_overlap

                # If there is overlap, combine the boxes
                if overlap_area > 0:
                    new_box['bbox'] = [
                        min(box1['bbox'][0], box2['bbox'][0]),
                        min(box1['bbox'][1], box2['bbox'][1]),
                        max(box1['bbox'][2], box2['bbox'][2]),
                        max(box1['bbox'][3], box2['bbox'][3])
                    ]
                    seen_indices.add(j)  # Mark the second box as seen

        combined_boxes.append(new_box)

    return combined_boxes


def draw_bounding_boxes(frame, ents):
    """
    Draw bounding boxes and labels on the frame.

    Parameters:
    - frame: The image/frame on which to draw.
    - ents: List of detected entities containing their bounding box, class, ID, and score.
    """
    for ent in ents:
        bbox = ent['bbox']
        class_label = ent['class']
        score = ent['score']
        detected_id = ent['id']
        x1, y1, x2, y2 = map(int, bbox)  # Convert coordinates to integers

        color = COLORS.get(class_label, (255, 0, 0))  # Default to red if class not found
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        label = f"{class_label.capitalize()} ID:{detected_id} {score:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, cv2.FILLED)
        cv2.putText(frame, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
