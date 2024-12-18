import cv2
import numpy as np
from PIL import ImageGrab
 
def approx_dino_contour(game_thresh):
    dino_img = cv2.imread('imgs/dino.png', cv2.IMREAD_GRAYSCALE)
    _, dino_thresh = cv2.threshold(dino_img, 127, 255,0)

    contours, _ = cv2.findContours(dino_thresh, 2, 1)
    dino_contour = contours[2]

    game_contours, _ = cv2.findContours(game_thresh, 2, 1)
    min_difference = float('inf')
    game_dino_contour = None
    for contour in game_contours:
        difference = cv2.matchShapes(dino_contour, contour, 1, 0.0)
        if difference < min_difference:
            game_dino_contour = contour
            min_difference = difference
    
    if min_difference < 0.1:
        return game_dino_contour
    return np.array([])


def nearest_obstacle_start(game_dino_contour, game_thresh, lowest_dino_height_percentage=0.4):
    reshaped_contour = np.reshape(game_dino_contour, (game_dino_contour.shape[0], game_dino_contour.shape[2]))
    right_most_point = max(reshaped_contour[:, 0])
    top_point = min(reshaped_contour[:, 1])
    bottom_point = max(reshaped_contour[:, 1])

    dino_height = abs(top_point - bottom_point)
    min_obstacle_x = right_most_point
    max_obstacle_y = top_point + int(dino_height * (1 - lowest_dino_height_percentage))

    modified_game_thresh = game_thresh.copy()
    modified_game_thresh[:, :min_obstacle_x] = 127
    modified_game_thresh[max_obstacle_y:, :] = 127
    obstacle_start_x = min(np.where(modified_game_thresh == 0)[1])
    return obstacle_start_x, modified_game_thresh


def capture_screen():
    screenshot = ImageGrab.grab()
    bgr_screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    resized_screenshot = cv2.resize(bgr_screenshot, (0, 0), fx = 1, fy = 1)
    return resized_screenshot


def revert_colors(thresh_img):
    black_pixels = np.where(thresh_img == 0)[0], np.where(thresh_img == 0)[1]
    white_pixels = np.where(thresh_img == 255)[0], np.where(thresh_img == 255)[1]
    thresh_img[black_pixels] = 255
    thresh_img[white_pixels] = 0
    return thresh_img


whole_screen = capture_screen()
# Select region of interest then press ENTER
roi = cv2.selectROI("ROI", whole_screen) 

while True:
    roi_screenshot = capture_screen()[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    grayscale_game_img = cv2.cvtColor(roi_screenshot, cv2.COLOR_BGR2GRAY)
    _, game_thresh = cv2.threshold(grayscale_game_img, 127, 255,0)
    if game_thresh[0, 0] == 0:
        game_thresh = revert_colors(game_thresh)
    game_dino_contour = approx_dino_contour(game_thresh)

    if len(game_dino_contour) > 1:
        obstacle_start_x, modified_game_thresh = nearest_obstacle_start(game_dino_contour, game_thresh)
        cv2.drawContours(game_thresh, [np.array([[[obstacle_start_x, 0]], [[obstacle_start_x, game_thresh.shape[1] - 1]]])], 0, 127, 3)
        cv2.drawContours(modified_game_thresh, [game_dino_contour], 0, 255, 2)
        cv2.imshow("Obstacle", modified_game_thresh)

    cv2.imshow("ROI", game_thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break