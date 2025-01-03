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

        self.is_ascending_to_top = False
        self.ascent_start_time = None
        self.secs_to_top_of_obstacle = 0.2

        self.is_over_obstacle = False
        self.secs_over_obstacle = 0.5
        self.over_obstacle_start_time = None


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
            obstacle_start_x = None

        self.prev_dino_contour = self.curr_dino_contour.copy()

        return obstacle_start_x

    
    def jump_now(self, nearest_obstacle):
        if not self.is_ascending_to_top and not self.is_over_obstacle:
            approx_obstacle_velocity = nearest_obstacle.velocity

            print(self.secs_over_obstacle)
            print(self.secs_to_top_of_obstacle)
            print()

            # approximate start of the obstacle after dino reaches its top
            adjusted_obstacle_end_x = nearest_obstacle.end_x - (approx_obstacle_velocity * self.secs_to_top_of_obstacle)
            left_most_point = self.calculate_dino_contour_properties(self.curr_dino_contour)["left_most_point"]
            if left_most_point > (adjusted_obstacle_end_x - (approx_obstacle_velocity * self.secs_over_obstacle)):
                self.is_ascending_to_top = True
                self.ascent_start_time = time.time()
                return True
        return False

    
    def update_jump_properties(self, nearest_obstacle):
        bottom_point = self.calculate_dino_contour_properties(self.curr_dino_contour)["bottom_point"]
        if bottom_point < nearest_obstacle.top_point:
            self.is_ascending_to_top = False
            if self.is_over_obstacle:
                self.secs_over_obstacle = time.time() - self.over_obstacle_start_time
            else:
                self.is_over_obstacle = True
                self.over_obstacle_start_time = time.time()
        else:
            self.is_over_obstacle = False

        if self.is_ascending_to_top:
            self.secs_to_top_of_obstacle = time.time() - self.ascent_start_time



class obstacle:
    def __init__(self):
        self.start_x = None
        self.end_x = None
        self.top_point = None
        
        self.prev_start_x = None
        self.last_recorded_velocity_time = 0
        self.velocity = 0

    def update_position_properties(self, dino, game_thresh):
        self.start_x = dino.nearest_obstacle_start(game_thresh)
        if self.start_x == None:
            self.start_x = game_thresh.shape[1] - 1
            self.end_x = self.start_x
            self.top_point = 0
        else:
            self.end_x = self.calculate_end_x(dino.obstacle_view_thresh, dino.get_game_dino_width())
            self.top_point = self.calculate_top_point(dino.max_obstacle_y, game_thresh)


    def calculate_end_x(self, obstacle_view_thresh, game_dino_width):
        curr_end_x = self.start_x
        while True:
            window_start = curr_end_x + 1
            window_end = window_start + game_dino_width
            obstacle_window_x_coors = np.where(obstacle_view_thresh[:, window_start:window_end] == 0)[1] + window_start
            if len(obstacle_window_x_coors) >= 1:
                curr_end_x = max(obstacle_window_x_coors)
            else:
                break
                
        return curr_end_x

    def calculate_top_point(self, min_top_point, game_thresh):
        curr_top_point = min_top_point
        while True:
            curr_top_point -= 1
            obstacle_window_row = game_thresh[curr_top_point, self.start_x:self.end_x]
            if len(np.where(obstacle_window_row ==0)[0]) == 0:
                break
        
        return curr_top_point

    def update_velocity(self):
        if self.prev_start_x == None:
            self.prev_start_x = self.start_x

        x_delta = max(0, self.prev_start_x - self.start_x)
        time_passed = time.time() - self.last_recorded_velocity_time

        self.prev_start_x = self.start_x
        self.last_recorded_velocity_time = time.time()

        self.velocity = x_delta / time_passed
        return self.velocity
          


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
nearest_obstacle = obstacle()

while True:
    roi_screenshot = capture_screen()[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    grayscale_game_img = cv2.cvtColor(roi_screenshot, cv2.COLOR_BGR2GRAY)
    _, game_thresh = cv2.threshold(grayscale_game_img, 127, 255,0)
    
    if game_thresh[0, 0] == 0:
        game_thresh = revert_colors(game_thresh)

    game_dino_contour = rex.approx_game_dino_contour(game_thresh)

    if game_dino_contour is not None:

        nearest_obstacle.update_position_properties(rex, game_thresh)
        if time.time() - nearest_obstacle.last_recorded_velocity_time > 1:
            nearest_obstacle.update_velocity()

        rex.update_jump_properties(nearest_obstacle)

        jump = rex.jump_now(nearest_obstacle)
        if jump:
            pyautogui.press('up')

        cv2.line(game_thresh, (nearest_obstacle.start_x, nearest_obstacle.top_point), (nearest_obstacle.start_x, (game_thresh.shape[0] - 1)), 127, 3)
        cv2.line(game_thresh, (nearest_obstacle.end_x, nearest_obstacle.top_point), (nearest_obstacle.end_x, (game_thresh.shape[0] - 1)), 127, 3)
        cv2.line(game_thresh, (nearest_obstacle.start_x, nearest_obstacle.top_point), (nearest_obstacle.end_x, nearest_obstacle.top_point), 127, 3)
        cv2.imshow("Obstacle View", rex.obstacle_view_thresh)
    
    cv2.imshow("ROI", game_thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break