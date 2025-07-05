import cv2
import numpy as np
import time
import uuid

class Obstacle:
    def __init__(self):
        self.start_x = None
        self.end_x = None
        self.top = None
        self.bottom = None
        
        self.pixels_per_sec = 0
        self.spawn_timestamp = None
        self.spawn_start_x = None
        self.velocity_dict = {"secs_passed_list":[], "displacement_list":[]}

        self.is_on_screen = True
        self.id = uuid.uuid4()


    def update_position_properties(self, game_thresh, obstacle_view_thresh, dino_width):
        self.start_x = self.calculate_start_x(obstacle_view_thresh)
        if self.start_x == None:
            self.is_on_screen = False
            return
        
        previous_end_x = self.end_x
        self.end_x = self.calculate_end_x(obstacle_view_thresh, dino_width)
        if previous_end_x != None and self.end_x > previous_end_x: # obstacle cannot move backwards
            self.is_on_screen = False
            return 
        
        self.top, self.bottom = self.calculate_top_and_bottom(game_thresh, obstacle_view_thresh)

    
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
    

    def calculate_top_and_bottom(self, game_thresh, obstacle_view_thresh):
        obstacle_view_window = obstacle_view_thresh[:, :self.end_x + 1]
        obstacle_y_coors = np.where(obstacle_view_window == 0)[0]
        obstacle_top_point = min(obstacle_y_coors)
        game_thresh_window = game_thresh[obstacle_top_point:, self.start_x : self.end_x + 1]
        game_window_black_ys = set(np.where(game_thresh_window == 0)[0])
        game_window_white_ys = set(np.where(game_thresh_window == 255)[0])
        set_difference = game_window_white_ys - game_window_black_ys
        obstacle_bottom_point = min(set_difference) + obstacle_top_point
        return obstacle_top_point, obstacle_bottom_point
    

    def update_pixels_per_sec(self):
        if self.is_on_screen:
            
            if self.spawn_timestamp == None:
                self.spawn_timestamp = time.time()
                self.spawn_start_x = self.start_x
                self.velocity_dict['secs_passed_list'] = [0]
                self.velocity_dict['displacement_list'] = [0]
            else:
                secs_passed = time.time() - self.spawn_timestamp
                displacement = self.spawn_start_x - self.start_x
                self.velocity_dict['secs_passed_list'].append(secs_passed)
                self.velocity_dict['displacement_list'].append(displacement)

            if len(self.velocity_dict['secs_passed_list']) > 1:
                x, y = self.velocity_dict['secs_passed_list'], self.velocity_dict['displacement_list']
                # best_fit_slope, _ = np.polyfit(x, y, deg=1)
                self.pixels_per_sec = y[-1] / x[-1]# best_fit_slope
                ##print(self.pixels_per_sec)    
    