import cv2
import numpy as np
from PIL import ImageGrab
import time

class dino():
    def __init__(self):
        self.min_obstacle_x = None
        self.min_dino_height_percentage = 0.4
        self.max_obstacle_y = None
        self.max_dino_height_percentage = 1.1
        self.min_obstacle_y = None
        self.prev_top_point = None
        self.last_off_gorund_time = 0
        self.model_dino_contour = self.generate_model_dino_contour()

    def generate_model_dino_contour(self): 
        model_dino_img = cv2.imread('imgs/dino.png', cv2.IMREAD_GRAYSCALE)
        model_dino_img = cv2.resize(model_dino_img, (0, 0), fx = 0.5, fy = 0.5)
        _, thresh = cv2.threshold(model_dino_img, 127, 255,0)
        contours, _ = cv2.findContours(thresh, 2, 1)
        model_dino_contour = contours[2]
        return model_dino_contour

    def approx_game_dino_contour(self, game_thresh):
        game_contours, _ = cv2.findContours(game_thresh, 2, 1)
        min_difference = float('inf')
        game_dino_contour = None
        for contour in game_contours:
            difference = cv2.matchShapes(self.model_dino_contour, contour, 1, 0.0)
            if difference < min_difference:
                game_dino_contour = contour
                min_difference = difference
        
        if min_difference < 0.1:
            return game_dino_contour
        return np.array([])


    def nearest_obstacle_start(self, game_dino_contour, game_thresh):
        reshaped_contour = np.reshape(game_dino_contour, (game_dino_contour.shape[0], game_dino_contour.shape[2]))
        right_most_point = max(reshaped_contour[:, 0])
        left_most_point = min(reshaped_contour[:, 0])
        dino_width = right_most_point - left_most_point
        top_point = min(reshaped_contour[:, 1])
        bottom_point = max(reshaped_contour[:, 1])
        dino_height = abs(top_point - bottom_point)
        
        if self.prev_top_point == None:
            self.prev_top_point = top_point
        y_position_delta = abs(self.prev_top_point - top_point)
        secs_on_ground = time.time() - self.last_off_gorund_time
        is_on_ground = y_position_delta == 0 and secs_on_ground > 0.1
        if is_on_ground:
            self.min_obstacle_x = right_most_point
            self.max_obstacle_y = top_point + int(dino_height * (1 - self.min_dino_height_percentage))
            self.min_obstacle_y = top_point - int(dino_height * (self.max_dino_height_percentage  - 1))
        else:
            self.last_off_gorund_time = time.time()

        obstacle_view_thresh = game_thresh.copy()
        obstacle_view_thresh[:, :self.min_obstacle_x] = 127
        obstacle_view_thresh[self.max_obstacle_y:, :] = 127
        obstacle_view_thresh[:self.min_obstacle_y, :] = 127

        obstacle_x_values = np.where(obstacle_view_thresh == 0)[1]
        if len(obstacle_x_values) >= 1:
            obstacle_start_x = min(obstacle_x_values)
        else:
            obstacle_start_x = game_thresh.shape[1] - 5

        self.prev_top_point = top_point
        
        cv2.drawContours(obstacle_view_thresh, [game_dino_contour], 0, 0, -1)
        cv2.line(game_thresh, (obstacle_start_x, 0), (obstacle_start_x, (game_thresh.shape[1] - 1)), 127, 3)
        cv2.imshow("Obstacle View", obstacle_view_thresh)

        return obstacle_start_x, dino_width

    
    def jump_now(self, game_thresh):
        game_dino_contour = self.approx_game_dino_contour(game_thresh)

        if len(game_dino_contour) > 1:
            obstacle_start_x, game_dino_width = self.nearest_obstacle_start(game_dino_contour, game_thresh)
            if obstacle_start_x - self.min_obstacle_x <= (game_dino_width * 0.5):
                return True

        return False


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

rex = dino()

while True:
    roi_screenshot = capture_screen()[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    grayscale_game_img = cv2.cvtColor(roi_screenshot, cv2.COLOR_BGR2GRAY)
    _, game_thresh = cv2.threshold(grayscale_game_img, 127, 255,0)
    
    if game_thresh[0, 0] == 0:
        game_thresh = revert_colors(game_thresh)
    
    jump = rex.jump_now(game_thresh)

    cv2.imshow("ROI", game_thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break