# importing all the needed libraries
import cv2
import numpy as np

# Finding Homography for the four corners in the ROI of the frame
pts_src = np.array([[574, 11], [762, 9], [1100, 246], [162, 254]])
pts_dst = np.array([[50, 0], [250, 0], [250, 500], [0, 500]])
H_Matrix, status = cv2.findHomography(pts_src, pts_dst)


# Function to detect the Hough lines
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    direction = extrapolate(line_img, lines)
    return direction, line_img


# Undistorting the image based on the camera parameters
def undistort_image(img):
    k = [[1.15422732e+03, 0.000000e+00, 6.71627794e+02], [0.000000e+00, 1.14818221e+03, 3.86046312e+02],
         [0.000000e+00, 0.000000e+00, 1.000000e+00]]
    k = np.array(k)
    d = [[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]]
    d = np.array(d)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(k, d, (w, h), 1, (w, h))
    dst = cv2.undistort(img, k, d, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


# Function to calculate mean of a list
def mean(list):
    return float(sum(list)) / max(len(list), 1)


# Function to detect the lanes from the given image
def detect_lanes(img):
    hsl_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # Creating a mask for yellow lines
    lower_mask_yellow = np.array([20, 120, 70], dtype='uint8')
    upper_mask_yellow = np.array([45, 200, 255], dtype='uint8')
    mask_yellow = cv2.inRange(hsl_img, lower_mask_yellow, upper_mask_yellow)
    yellow_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_yellow).astype(np.uint8)

    # Creating a mask for white lines
    lower_mask_white = np.array([0, 200, 0], dtype='uint8')
    upper_mask_white = np.array([255, 255, 255], dtype='uint8')
    mask_white = cv2.inRange(hsl_img, lower_mask_white, upper_mask_white)
    white_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_white).astype(np.uint8)

    # Adding both the detected white and yellow lines
    lanes = cv2.bitwise_or(yellow_detect, white_detect)
    new_lanes = cv2.cvtColor(lanes, cv2.COLOR_HLS2BGR)
    final = cv2.cvtColor(new_lanes, cv2.COLOR_BGR2GRAY)

    return final


# Function to extrapolate the lines and draw the bounding region between the lanes
def extrapolate(img, lines, color=[0, 255, 0], thickness=12):
    straight = "straight"
    right = "right"
    left = "left"

    if lines is None:
        return

    ymin_global = img.shape[0]
    ymax_global = img.shape[0]

    # left lane line variables
    all_left_grad = []
    all_left_y = []
    all_left_x = []

    # right lane line variables
    all_right_grad = []
    all_right_y = []
    all_right_x = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)

            if (gradient > 0):
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]

    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)

    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)

    if (len(all_left_grad) > 0) and (len(all_right_grad) > 0):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        cv2.line(img, (upper_left_x, ymin_global),
                 (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global),
                 (lower_right_x, ymax_global), color, thickness)

        pts = np.array([[upper_left_x, ymin_global], [upper_right_x, ymin_global], [lower_right_x, ymax_global],
                        [lower_left_x, ymax_global]], np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.fillConvexPoly(img, pts, (100, 100, 100), lineType=8, shift=0)

        if int(upper_right_x) in range(40, 53):
            return straight
        elif int(upper_right_x > 50):
            return right
        elif int(upper_right_x < 30):
            return left


# Function where all the image operations happen
def image_process(img):

    # Creating an ROI rejecting the sky region
    crop_img = img[480:720, 0:1280]

    # Undistorting the image
    nice_img = undistort_image(crop_img)

    # Detecting white and yellow lines by creating a mask
    white = detect_lanes(nice_img)

    # Smoothing the image by using a Gaussian Filter
    blur = cv2.GaussianBlur(white, (5, 5), 0)

    # Warping the image to get the bird's view of the road
    im_out = cv2.warpPerspective(blur, H_Matrix, (300, 600))

    # Detecting the edges in the image
    edges = cv2.Canny(im_out, 100, 200)

    # Detecting the lines in the image by using Hough Transform
    rho = 2
    theta = np.pi / 180
    # threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 100
    min_line_len = 150
    max_line_gap = 800
    direction, line_image = hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)

    # Rewarping the image after the image processing is done
    out = cv2.warpPerspective(line_image, np.linalg.inv(H_Matrix), (1209, 175))

    # # Image Addition
    rows, cols, channels = out.shape
    roi = nice_img[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask
    img2gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    # add a threshold
    ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY_INV)

    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2_fg = cv2.bitwise_and(out, out, mask=mask_inv)

    Final = cv2.add(img1_bg, img2_fg)

    cv2.putText(Final, direction, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Final_output", Final)


video = cv2.VideoCapture('challenge_video.mp4')
while video.isOpened():
    opened, frame = video.read()

    if opened:
        image_process(frame)
        cv2.waitKey(1)
    else:
        break
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
video.release()
