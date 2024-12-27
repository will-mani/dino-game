import cv2
import numpy as np
from PIL import ImageGrab
import time
import pyautogui

class dino:
    def __init__(self):
        self.min_obstacle_x = None
        self.obstacle_min_dino_height_percentage = 0.4

        self.max_obstacle_y = None
        self.obstacle_max_dino_height_percentage = 1.1

        self.min_obstacle_y = None
        self.last_off_gorund_time = 0

        self.model_dino_contour = self.generate_model_dino_contour()
        self.curr_dino_contour = None
        self.prev_dino_contour = None

        self.obstacle_view_thresh = None


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
        for contour in game_contours:
            difference = cv2.matchShapes(self.model_dino_contour, contour, 1, 0.0)
            if difference < min_difference:
                self.curr_dino_contour = contour
                min_difference = difference
        
        if min_difference > 0.1:
            self.curr_dino_contour = None

        return self.curr_dino_contour


    def calculate_dino_contour_properties(self, dino_contour):
        properties_dict = {}
        reshaped_contour = np.reshape(dino_contour, (dino_contour.shape[0], dino_contour.shape[2]))
        
        properties_dict["right_most_point"] = max(reshaped_contour[:, 0])
        properties_dict["left_most_point"] = min(reshaped_contour[:, 0])
        properties_dict["dino_width"] = properties_dict["right_most_point"] - properties_dict["left_most_point"]
        
        properties_dict["top_point"] = min(reshaped_contour[:, 1])
        properties_dict["bottom_point"] = max(reshaped_contour[:, 1])
        properties_dict["dino_height"] =  properties_dict["bottom_point"] - properties_dict["top_point"]

        return properties_dict

    
    def get_game_dino_width(self):
        game_dino_properties_dict = self.calculate_dino_contour_properties(self.curr_dino_contour)
        return game_dino_properties_dict["dino_width"]


    def is_on_ground(self):
        curr_top_point = self.calculate_dino_contour_properties(self.curr_dino_contour)["top_point"]
        if self.prev_dino_contour is None:
            prev_top_point = curr_top_point
        else:
            prev_top_point = self.calculate_dino_contour_properties(self.prev_dino_contour)["top_point"] 
        y_position_delta = abs(prev_top_point - curr_top_point)
        secs_on_ground = time.time() - self.last_off_gorund_time
        return y_position_delta == 0 and secs_on_ground > 1


    def update_obstacle_view(self, game_thresh):
        self.obstacle_view_thresh = game_thresh.copy()
        self.obstacle_view_thresh[:, :self.min_obstacle_x] = 127
        self.obstacle_view_thresh[self.max_obstacle_y:, :] = 127
        self.obstacle_view_thresh[:self.min_obstacle_y, :] = 127
        cv2.drawContours(self.obstacle_view_thresh, [self.curr_dino_contour], 0, 255, -1)


    def nearest_obstacle_start(self, game_thresh):
        game_dino_properties_dict = self.calculate_dino_contour_properties(self.curr_dino_contour)
        right_most_point = game_dino_properties_dict["right_most_point"]
        top_point = game_dino_properties_dict["top_point"]
        dino_height = game_dino_properties_dict["dino_height"]
        
        if self.is_on_ground():
            self.min_obstacle_x = right_most_point
            self.max_obstacle_y = top_point + int(dino_height * (1 - self.obstacle_min_dino_height_percentage))
            self.min_obstacle_y = top_point - int(dino_height * (self.obstacle_max_dino_height_percentage  - 1))
        else:
            self.last_off_gorund_time = time.time()

        self.update_obstacle_view(game_thresh)

        obstacle_x_coors = np.where(self.obstacle_view_thresh == 0)[1]
        if len(obstacle_x_coors) >= 1:
            obstacle_start_x = min(obstacle_x_coors)
        else:
            obstacle_start_x = game_thresh.shape[1] - 1

        self.prev_dino_contour = self.curr_dino_contour.copy()

        return obstacle_start_x

    
    def jump_now(self, nearest_obstacle):
        if ((nearest_obstacle.start_x - self.min_obstacle_x) >= 0 and
            (nearest_obstacle.start_x - self.min_obstacle_x) <= (self.get_game_dino_width() * 3)):
            return True

        return False



class obstacle:
    def __init__(self, obstacle_start_x, obstacle_view_thresh, game_dino_width):
        self.start_x = obstacle_start_x
        self.end_x = self.calculate_end_x(obstacle_view_thresh, game_dino_width)


    def calculate_end_x(self, obstacle_view_thresh, game_dino_width):
        curr_end_x = self.start_x
        while True:
            window_start = curr_end_x + 1
            window_end = window_start + game_dino_width
            window_obstacle_x_coors = np.where(obstacle_view_thresh[:, window_start:window_end] == 0)[1] + window_start
            if len(window_obstacle_x_coors) >= 1:
                curr_end_x = max(window_obstacle_x_coors)
            else:
                break
                
        return curr_end_x



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
nearest_obstacle = None

while True:
    roi_screenshot = capture_screen()[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    grayscale_game_img = cv2.cvtColor(roi_screenshot, cv2.COLOR_BGR2GRAY)
    _, game_thresh = cv2.threshold(grayscale_game_img, 127, 255,0)
    
    if game_thresh[0, 0] == 0:
        game_thresh = revert_colors(game_thresh)

    game_dino_contour = rex.approx_game_dino_contour(game_thresh)

    if game_dino_contour is not None:

        obstacle_start_x= rex.nearest_obstacle_start(game_thresh)
        nearest_obstacle = obstacle(obstacle_start_x, rex.obstacle_view_thresh, rex.get_game_dino_width())

        jump = rex.jump_now(nearest_obstacle)
        if jump:
            pyautogui.press('up')

        cv2.line(game_thresh, (nearest_obstacle.start_x, 0), (nearest_obstacle.start_x, (game_thresh.shape[0] - 1)), 127, 3)
        cv2.line(game_thresh, (nearest_obstacle.end_x, 0), (nearest_obstacle.end_x, (game_thresh.shape[0] - 1)), 127, 3)
        cv2.imshow("Obstacle View", rex.obstacle_view_thresh)
    
    cv2.imshow("ROI", game_thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break