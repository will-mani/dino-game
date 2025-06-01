import cv2
import numpy as np
from PIL import ImageGrab
import time
import pyautogui
from pynput import keyboard

class dino:
    def __init__(self):
        self.right_most_point = None
        self.left_most_point = None
        self.width = None

        self.prev_top_point = None
        self.top_point = None
        self.bottom_point = None
        self.height = None

        self.model_contour = self.generate_model_contour()
        self.curr_contour = None

        self.last_off_gorund_timestamp = 0
        self.close_to_ground_min_y = None
        self.close_to_ground_dino_height_percentage = 0.5
        self.is_close_to_ground = True
    
        self.obstacle_view_min_x= None
        self.obstacle_view_min_dino_height_percentage = 0.4
        self.obstacle_view_max_y = None
        self.obstacle_view_max_dino_height_percentage = 1.2
        self.obstacle_view_min_y = None

        self.obstacle_view_thresh = None

        self.is_ascending_to_min_y = False
        self.ascent_start_timestamp = None
        self.secs_to_top_of_min_y = 0
        # min_y is short for obstacle_view_min_y


    def generate_model_contour(self): 
        model_dino_img = cv2.imread('imgs/dino.png', cv2.IMREAD_GRAYSCALE)
        model_dino_img = cv2.resize(model_dino_img, (0, 0), fx = 0.5, fy = 0.5)
        _, thresh = cv2.threshold(model_dino_img, 127, 255,0)
        contours, _ = cv2.findContours(thresh, 2, 1)
        model_dino_contour = contours[2]
        return model_dino_contour


    def approx_game_contour(self, game_thresh):
        game_contours, _ = cv2.findContours(game_thresh, 2, 1)
        min_difference = float('inf')
        for potential_contour in game_contours:
            difference = cv2.matchShapes(self.model_contour, potential_contour, 1, 0.0)
            if difference < min_difference:
                self.curr_contour = potential_contour
                min_difference = difference
        
        if min_difference > 0.1:
            self.curr_contour = None

        return self.curr_contour
    
    def update_contour_properties(self):
        reshaped_contour = np.reshape(self.curr_contour, (self.curr_contour.shape[0], self.curr_contour.shape[2]))
        
        self.right_most_point = max(reshaped_contour[:, 0])
        self.left_most_point = min(reshaped_contour[:, 0])
        self.width = self.right_most_point - self.left_most_point

        self.prev_top_point = self.top_point
        self.top_point = min(reshaped_contour[:, 1])
        self.bottom_point = max(reshaped_contour[:, 1])
        self.height = self.bottom_point - self.top_point


    def is_defenitely_on_ground(self):
        if self.prev_top_point == None:
            self.prev_top_point = self.top_point
        y_position_delta = abs(self.prev_top_point - self.top_point)
        if y_position_delta > 0:
            self.last_off_gorund_timestamp = time.time()
            return False
        secs_on_ground = time.time() - self.last_off_gorund_timestamp
        return secs_on_ground > 2
    
    
    def update_obstacle_view_properties(self, game_thresh):
        if self.is_defenitely_on_ground():
            self.obstacle_view_min_x = self.right_most_point
            self.obstacle_view_max_y = self.bottom_point - int(self.height * self.obstacle_view_min_dino_height_percentage)
            self.obstacle_view_min_y = self.bottom_point - int(self.height * self.obstacle_view_max_dino_height_percentage)
            self.close_to_ground_min_y = self.bottom_point - int(self.height * self.close_to_ground_dino_height_percentage)

        self.obstacle_view_thresh = game_thresh.copy()
        self.obstacle_view_thresh[:, :self.obstacle_view_min_x] = 127
        self.obstacle_view_thresh[self.obstacle_view_max_y:, :] = 127
        self.obstacle_view_thresh[:self.obstacle_view_min_y, :] = 127
        cv2.drawContours(self.obstacle_view_thresh, [self.curr_contour], 0, 255, -1)


    def jump_now(self, nearest_obstacle):
        if (not self.is_ascending_to_min_y and self.is_close_to_ground 
            and self.right_most_point < nearest_obstacle.start_x and self.top_point < nearest_obstacle.bottom):

            adjusted_obstacle_start = nearest_obstacle.start_x * 0.8
            adjusted_obstacle_velocity = nearest_obstacle.pixels_per_sec * 1.25

            # print(adjusted_obstacle_velocity)

            if (adjusted_obstacle_start - (adjusted_obstacle_velocity * self.secs_to_top_of_min_y)) <= self.right_most_point:
                return True

        return False

    
    def update_jump_properties(self):
        if self.bottom_point >= self.close_to_ground_min_y:
            self.is_close_to_ground = True
        else:
            self.is_close_to_ground = False
    
        if self.bottom_point < self.obstacle_view_min_y:
            self.is_ascending_to_min_y = False

        if self.is_ascending_to_min_y:
            self.secs_to_top_of_min_y = time.time() - self.ascent_start_timestamp



class obstacle:
    def __init__(self):
        self.start_x = None
        self.end_x = None
        self.top = None
        self.bottom = None
        
        self.pixels_per_sec = 0
        self.can_update_velocity = True

        self.is_visible = False
        self.last_visible_end_x = float('inf')
        self.respawn_timestamp = None
        self.respawn_start_x = None
        self.velocity_dict = {"secs_passed_list":[], "displacement_list":[]}


    def update_position_properties(self, dino, game_thresh):
        self.start_x = self.calculate_start_x(dino.obstacle_view_thresh)
        if self.start_x == None:
            self.is_visible = False
            self.start_x = game_thresh.shape[1] - 1
            self.end_x = self.start_x
            self.top = 0
            self.bottom = game_thresh.shape[0] - 1
        else:
            self.is_visible = True
            self.end_x = self.calculate_end_x(dino.obstacle_view_thresh, dino.width)
            self.top, self.bottom = self.calculate_top_and_bottom(dino.obstacle_view_thresh)

    
    def calculate_start_x(self, obstacle_view_thresh):
        obstacle_x_coors = np.where(obstacle_view_thresh == 0)[1]
        if len(obstacle_x_coors) >= 1:
            obstacle_start_x = min(obstacle_x_coors)
        else:
            obstacle_start_x = None

        return obstacle_start_x
    

    def calculate_end_x(self, obstacle_view_thresh, dino_width):
        curr_end_x = self.start_x
        while True:
            window_start = curr_end_x + 1
            window_end = window_start + dino_width
            obstacle_window_x_coors = np.where(obstacle_view_thresh[:, window_start:window_end] == 0)[1] + window_start
            if len(obstacle_window_x_coors) >= 1:
                curr_end_x = max(obstacle_window_x_coors)
            else:
                break
                
        return curr_end_x
    

    def calculate_top_and_bottom(self, obstacle_view_thresh):
        window_end = self.end_x + 1
        obstacle_view_window = obstacle_view_thresh[:, :window_end]
        obstacle_y_coors = np.where(obstacle_view_window == 0)[0]
        obstacle_top_point = min(obstacle_y_coors)
        obstacle_bottom_point = max(obstacle_y_coors)
        return obstacle_top_point, obstacle_bottom_point
    

    def update_pixels_per_sec(self):
        if self.is_visible:
            if self.end_x > self.last_visible_end_x and len(self.velocity_dict['secs_passed_list']) > 1:
                x, y = self.velocity_dict['secs_passed_list'], self.velocity_dict['displacement_list']
                # best_fit_slope, _ = np.polyfit(x, y, deg=1)
                self.pixels_per_sec = y[-1] / x[-1]# best_fit_slope
                print(self.pixels_per_sec)

            if self.end_x > self.last_visible_end_x or self.respawn_timestamp == None:
                self.respawn_timestamp = time.time()
                self.respawn_start_x = self.start_x
                self.velocity_dict['secs_passed_list'] = [0]
                self.velocity_dict['displacement_list'] = [0]
            else:
                secs_passed = time.time() - self.respawn_timestamp
                displacement = self.respawn_start_x - self.start_x
                self.velocity_dict['secs_passed_list'].append(secs_passed)
                self.velocity_dict['displacement_list'].append(displacement)

            self.last_visible_end_x = self.end_x

global jump_count
jump_count = 0


def capture_screen(image_grab_bbox):
    screenshot = ImageGrab.grab(bbox=image_grab_bbox)
    bgr_screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    resized_screenshot = cv2.resize(bgr_screenshot, (0, 0), fx = 1, fy = 1)
    return resized_screenshot


def revert_colors(thresh_img):
    black_pixels = np.where(thresh_img == 0)[0], np.where(thresh_img == 0)[1]
    white_pixels = np.where(thresh_img == 255)[0], np.where(thresh_img == 255)[1]
    thresh_img[black_pixels] = 255
    thresh_img[white_pixels] = 0
    return thresh_img


def on_press(key):
    global jump_count
    
    if key == keyboard.Key.up or key == keyboard.Key.space:
        print("Jump Time!")
        rex.is_ascending_to_min_y = True
        rex.ascent_start_timestamp = time.time()

        jump_count += 1


keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()


whole_screen = capture_screen(image_grab_bbox=None)
# Select region of interest then press ENTER
roi = cv2.selectROI("ROI", whole_screen) 

rex = dino()
nearest_obstacle = obstacle()

auto_pilot = False # human starts game off in control to set: obstacle.pixels_per_sec and dino.secs_to_top_of_min_y

# frame_count = 0
# sec_start = time.time()
while True:
    # if time.time() - sec_start >= 1:
    #     print(frame_count)
    #     frame_count = 0
    #     sec_start = time.time()
    # else:
    #     frame_count += 1

    roi_screenshot = capture_screen(image_grab_bbox=(int(roi[0]), int(roi[1]), int(roi[0]+roi[2]), int(roi[1]+roi[3])))
    grayscale_game_img = cv2.cvtColor(roi_screenshot, cv2.COLOR_BGR2GRAY)
    _, game_thresh = cv2.threshold(grayscale_game_img, 127, 255,0)
    
    if game_thresh[0, 0] == 0:
        game_thresh = revert_colors(game_thresh)

    game_dino_contour = rex.approx_game_contour(game_thresh)

    if game_dino_contour is not None:

        rex.update_contour_properties()
        rex.update_obstacle_view_properties(game_thresh)

        nearest_obstacle.update_position_properties(rex, game_thresh)
        nearest_obstacle.update_pixels_per_sec()

        rex.update_jump_properties()

        if auto_pilot:
            if rex.jump_now(nearest_obstacle):
                pyautogui.press('up')
                pyautogui.keyUp('up')

        cv2.line(game_thresh, (nearest_obstacle.start_x, nearest_obstacle.top), (nearest_obstacle.start_x, nearest_obstacle.bottom), 127, 3)
        cv2.line(game_thresh, (nearest_obstacle.end_x, nearest_obstacle.top), (nearest_obstacle.end_x, nearest_obstacle.bottom), 127, 3)
        cv2.line(game_thresh, (nearest_obstacle.start_x, nearest_obstacle.top), (nearest_obstacle.end_x, nearest_obstacle.top), 127, 3)
        cv2.line(game_thresh, (0, rex.obstacle_view_min_y), (game_thresh.shape[1], rex.obstacle_view_min_y), 0, 1)

        cv2.imshow("Obstacle View", rex.obstacle_view_thresh)
    
    cv2.imshow("ROI", game_thresh)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # enabling auto_pilot
    elif k == ord('a'):
        auto_pilot = True
        print("Auto Pilot Enabled.")

    # reset after dying
    elif k == ord('r'):
        rex = dino()
        print("Dino Re-initialized!")
        auto_pilot = False
        print("Auto Pilot off.")
        jump_count = 0

    if jump_count == 3:
        auto_pilot = True
        print("Three man(ual) jumps complete.\nAuto Pilot Enabled.")