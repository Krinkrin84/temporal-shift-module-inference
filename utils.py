import random
import cv2


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov7 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    an opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color_dict = {
        'normal': (0, 255, 0),  # Green
        'shaking': (0, 0, 255),  # Red
        'hitting': (0, 0, 255),  # Red
        'suspicious': (0, 255, 255)  # Yellow
    }
    if not color:
        color = color_dict.get(label[:-5], [random.randint(0, 255) for _ in range(3)])
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_result(frame, alarm_status, result_boxes, result_scores, result_classid, categories):
    #        if self.GPIO_socket_thread.checkBuzzerStatus():
    #            self.GPIO_socket_thread.GPIO_output.toggleBuzzer()

    # if self.GPIO_socket_thread.checkResetStatus() == True:
    #     self.drown_analyser.reset_count()
    #
    # if self.GPIO_socket_thread.checkResetButtonStatus() == True:
    #     if is_GPIO_host:
    #         for z, ips in SocketGPIOThread.zones.items():
    #             for ip in ips:
    #                 self.GPIO_output.sendTo("reset", ip)

    # Draw status text
    if alarm_status == 0:
        # green light
        cv2.putText(
            frame,
            "Normal",
            (100, 100),
            0,
            2,
            (0, 255, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        # self.GPIO_output.send(0)
        # print(f"STATUS 0")
    elif alarm_status == 1:
        # red light
        cv2.putText(
            frame,
            "Suspicious...",
            (100, 100),
            0,
            2,
            (0, 255, 255),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        # self.GPIO_output.send(1)
        # print(f"STATUS 1")
    elif alarm_status == 2:
        # red light
        cv2.putText(
            frame,
            "Warning!!",
            (100, 100),
            0,
            2,
            (0, 0, 255),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        # self.GPIO_output.send(2)
        # print(f"STATUS 2")
    else:
        raise NotImplementedError

    # Draw rectangles and labels on the original image
    for j in range(len(result_boxes)):
        box = result_boxes[j]
        plot_one_box(box, frame, label="{}:{:.2f}".format(categories[int(result_classid[j])], result_scores[j]))

    return frame


def norm_brightness(frame, val=125):
    # Splitting into HSV
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

    # Normalizing the brightness
    v = cv2.normalize(v, None, alpha=0, beta=val, norm_type=cv2.NORM_MINMAX)

    # Convert back into HSV
    hsv = cv2.merge((h, s, v))

    # Convert into color img
    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Return coloured image
    return res


def get_combined_squared_bounding_box(boxes):
    min_x, min_y, max_x, max_y = boxes[0]

    # Find the minimum and maximum coordinates
    for box in boxes[1:]:
        x1, y1, x2, y2 = box
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    # Calculate width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y
    # # Check if width to height ratio is smaller than 1:2
    if width * 2 < height:
        height /= 2  # Crop the top half

        min_y -= (height / 2 - height / 20)
        max_y -= (height / 2 - height / 20)

    # Calculate the combined bounding box dimensions
    half_side_length = max(width, height) // 2

    center_x, center_y = (max_x + min_x) // 2, (max_y + min_y) // 2

    return int(center_x - half_side_length), int(center_y - half_side_length), int(center_x + half_side_length), int(
        center_y + half_side_length)
