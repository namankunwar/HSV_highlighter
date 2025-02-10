import os
import pickle
import time
import PIL.Image, PIL.ImageTk
import numpy as np
import cv2
import tkinter as tk
from PIL import ImageTk, Image
import yaml
import glob
from pathlib import Path
from tkinter.filedialog import askdirectory


FILE = Path(__file__).parent
INPUT = Path(FILE / r'C:\Naman\Learning\yolov5\runs\detect\exp5\crops\bottle')

# PICKLE_DIR = os.path.join()

class Imager:
    # Define the necessary variables
    tolerance = 1
    listG = []
    fg_range = [] 
    all_images_list = []    
    drawing = False
    erase = False
    color_green = (0, 0, 255) 
    alpha = 0.9
    beta = 0.9
    gamma = 0.9
    current_index = 0
    current_original_image = 0 
    current_mask_image = 0
    scale_percent = 50
    directory = FILE
    diff_thresh = 30
    
    
    def __init__(self, directory):
        self.pickle_path = os.path.join(FILE, r'C:\Naman\Learning\yolov5\hsv_highlighter.pkl')
        self.zoom_level = 1.0  # Initialize zoom level
        self.zoom_x, self.zoom_y = 0, 0  # Initialize zoom center
        self.multiple_masks = self.load_from_pickle_file(self.pickle_path)
        # Save the initial empty list to the pickle file
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(self.multiple_masks, f)
        
        self.load_all_images(directory)
        self.load_image()
    
    def load_all_images(self,dir_path ):
        for image_path in os.listdir(dir_path):
            if '.jpg' in image_path:
                self.all_images_list.append(os.path.join(dir_path, str(image_path)))
    
    def load_image(self):        
        if self.current_original_image is not None:
            self.current_original_image = cv2.imread(self.all_images_list[self.current_index])
            self.current_mask_image = np.zeros_like(self.current_original_image[:, :, 0])
            width = int(self.current_original_image.shape[1] * self.scale_percent / 100)
            height = int(self.current_original_image.shape[0] * self.scale_percent / 100)
            dim = (width, height)
            self.current_original_image = cv2.resize(self.current_original_image,dim, cv2.INTER_AREA)
            self.img_copied = self.current_original_image.copy()
        else:
            pass
        
    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.all_images_list)
        self.load_image()
        self.current_mask_image, self.mask_img_fg = self.highlighter(self.multiple_masks, self.color_green)
        self.current_original_image = cv2.addWeighted(self.current_original_image, self.alpha, self.mask_img_fg, self.beta, self.gamma)
        self.refresh()

    def previous_image(self):
        self.current_index = (self.current_index - 1 + len(self.all_images_list)) % len(self.all_images_list)
        self.load_image()
        self.current_mask_image, self.mask_img_fg = self.highlighter(self.multiple_masks, self.color_green)
        self.current_original_image = cv2.addWeighted(self.current_original_image, self.alpha, self.mask_img_fg, self.beta, self.gamma)
        self.refresh() 
    
    
    def change_tolerance(self, value): # tolerance in trackbar
        self.tolerance = value if value != 0 else 1

    def hsv_ranging(self, hsv_values): # use tolerance for combination of hsv values. 
                                       # eg: [[7,7,7], [9,9,9]] -> here tolerence=1 for [8,8,8]
        last_hsv_value = hsv_values[-1]
        all_ranges = []
        lower_range = (last_hsv_value[0] - self.tolerance, last_hsv_value[1] - self.tolerance, last_hsv_value[2] - self.tolerance)
        upper_range = (last_hsv_value[0] + self.tolerance, last_hsv_value[1] + self.tolerance, last_hsv_value[2] + self.tolerance)
        ranges = []
        for i in range(lower_range[0], upper_range[0] + 1):
            for j in range(lower_range[1], upper_range[1] + 1):
                for k in range(lower_range[2], upper_range[2] + 1):
                    ranges.append((i, j, k))
        all_ranges.extend(ranges)
        unique_ranges = np.unique(np.array([sub_list for sub_list in all_ranges]),axis=0)
        unique_ranges = unique_ranges.tolist()
        all_ranges = [t for t in all_ranges if all(i >= 0 for i in t) if t[0] <= 179 if t[1] <= 255 if t[2] <= 255]
        unique_ranges = [t for t in unique_ranges if all(i >= 0 for i in t) if t[0] <= 179 if t[1] <= 255 if t[2] <= 255]
        masks = [min(unique_ranges), max(unique_ranges)]

        return masks
    
    def hsv_comparing_with_new_mask(self, masks, mask_list):
        append_new_masks = 0
        low_h1, low_s1, low_v1 = masks[0][0], masks[0][1], masks[0][2]
        high_h1, high_s1, high_v1 = masks[1][0], masks[1][1], masks[1][2]        

        if len(mask_list) > 0:
            for item in mask_list:
                # Ensure item[0] is a list (not a tuple) for modification
                if isinstance(item[0], tuple):
                    item[0] = list(item[0])  # Convert tuple to list if needed
                if isinstance(item[1], tuple):
                    item[1] = list(item[1])  # Convert tuple to list if needed

                low_h, low_s, low_v = item[0][0], item[0][1], item[0][2]
                high_h, high_s, high_v = item[1][0], item[1][1], item[1][2]

                compare_low = lambda low_x, low_x1: low_x1 if (low_x - low_x1) < self.diff_thresh and (low_x - low_x1) > 0 else low_x
                compare_high = lambda high_x, high_x1: high_x if (high_x - high_x1) < self.diff_thresh and (high_x - high_x1) > 0 else high_x1

                if (abs(low_h - low_h1) < self.diff_thresh and abs(high_h - high_h1) < self.diff_thresh 
                    and abs(low_s - low_s1) < self.diff_thresh and abs(high_s - high_s1) < self.diff_thresh 
                    and abs(low_v - low_v1) < self.diff_thresh and abs(high_v - high_v1) < self.diff_thresh):

                    low_h = compare_low(low_h, low_h1)
                    low_s = compare_low(low_s, low_s1)
                    low_v = compare_low(low_v, low_v1)

                    high_h = compare_high(high_h, high_h1)
                    high_s = compare_high(high_s, high_s1)
                    high_v = compare_high(high_v, high_v1)

                    item[0][0], item[0][1], item[0][2] = low_h, low_s, low_v
                    item[1][0], item[1][1], item[1][2] = high_h, high_s, high_v 
                    append_new_masks += 1

        if append_new_masks == 0:
            mask_list.append(masks)

        return mask_list
    
    def hsv_comparing_within_the_list(self, mask_list):
        """
        Merges overlapping or closely matching HSV (Hue, Saturation, Value) ranges in a list based on a predefined threshold.
        """

        from copy import deepcopy

        low, high = 0, 1  # Indices for low and high HSV bounds
        h, s, v = 0, 1, 2  # Indices for Hue, Saturation, and Value components

        index_to_remove = set()  # Use a set to store unique indices to remove
        temp_lis = deepcopy(mask_list)  # Create a copy to avoid modifying the original list during iteration

        # Iterate over each HSV range in the list
        for i_index, i in enumerate(temp_lis):
            low_h1, low_s1, low_v1 = i[low][h], i[low][s], i[low][v]
            high_h1, high_s1, high_v1 = i[high][h], i[high][s], i[high][v]

            for j_index in range(i_index + 1, len(temp_lis)):
                if j_index in index_to_remove:
                    continue  # Skip already marked indices
                
                j = temp_lis[j_index]
                low_h2, low_s2, low_v2 = j[low][h], j[low][s], j[low][v]
                high_h2, high_s2, high_v2 = j[high][h], j[high][s], j[high][v]

                if (abs(low_h1 - low_h2) < self.diff_thresh and abs(high_h1 - high_h2) < self.diff_thresh and
                    abs(low_s1 - low_s2) < self.diff_thresh and abs(high_s1 - high_s2) < self.diff_thresh and
                    abs(low_v1 - low_v2) < self.diff_thresh and abs(high_v1 - high_v2) < self.diff_thresh):

                    # Merge ranges
                    i[low] = (min(low_h1, low_h2), min(low_s1, low_s2), min(low_v1, low_v2))
                    i[high] = (max(high_h1, high_h2), max(high_s1, high_s2), max(high_v1, high_v2))

                    index_to_remove.add(j_index)  # Mark for removal

        # Remove marked indices **in reverse order** to prevent shifting issues
        for ix in sorted(index_to_remove, reverse=True):
            if 0 <= ix < len(temp_lis):  # Ensure index exists
                del temp_lis[ix]

        return temp_lis  # No recursive call needed

    
    def highlighter(self, mask_list, color): # masking the ranges from self.multiple using cv2.inRange
        
        # masks = [list(tuple(j) for j in i) for i in set(tuple(tuple(i) for i in j) for j in mask_list)]
        masks = [inner_list for i, inner_list in enumerate(mask_list) if inner_list not in mask_list[:i]]
        combined_mask = np.zeros_like(self.img_copied[:, :, 0])
        for mask in masks:
            low = np.array(mask[0])
            up = np.array(mask[1])
            hsv_image = cv2.cvtColor(self.img_copied, cv2.COLOR_BGR2HSV)
            temp_mask = cv2.inRange(hsv_image, low, up)
            combined_mask = cv2.bitwise_or(combined_mask, temp_mask)
        green_image = np.zeros_like(self.img_copied)
        green_image[:] = color
        masked_image = cv2.bitwise_and(self.img_copied, self.img_copied, mask=combined_mask)
        green_image = cv2.bitwise_or(green_image, green_image, mask=combined_mask)
        
        return masked_image, green_image

    def load_from_pickle_file(self, filename): # load self.multiple_masks from pickle
        try:
            with open(filename, 'rb') as f1:
                obj1 = pickle.load(f1)
            print("Segmentation data loaded successfully.")
            return obj1
        except FileNotFoundError:

            print("No previous segmentation data found. Starting fresh.")
            return []  # Return an empty list if no file exists
        

    def append_to_pickle_file(self, mask_list): # append self.multiple_masks in pickle
        
        # masks = [list(tuple(j) for j in i) for i in set(tuple(tuple(i) for i in j) for j in mask_list)]
        masks = [inner_list for i, inner_list in enumerate(mask_list) if inner_list not in mask_list[:i]]
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(masks, f)
        print("Segmentation data saved successfully.")

    def hsv_(self, list_hsv):
        if len(self.listG) < 0:
            pass
        else:
            self.listG.append(list_hsv)
            self.listG = [inner_list for i, inner_list in enumerate (self.listG) if inner_list not in self.listG[:i]] 

    @staticmethod
    def hsv_finder(output, y, x):
        colors = output[y, x]
        colors = np.array([[[colors[0], colors[1], colors[2]]]])
        hsv = cv2.cvtColor(colors, cv2.COLOR_BGR2HSV)
        hsvlist = hsv.tolist()
        h = hsvlist[0][0][0]
        s = hsvlist[0][0][1]
        v = hsvlist[0][0][2]
        return h, s, v
    
    def label_images(self):
        original_canvas = cv2.resize(self.img_copied,(255,255))
        original_canvas = cv2.cvtColor(original_canvas , cv2.COLOR_BGR2RGB)
        masked_resized_2_canvas = cv2.resize(self.current_original_image, (255, 255))
        masked_resized_2_canvas = cv2.cvtColor(masked_resized_2_canvas, cv2.COLOR_BGR2RGB)
        original_photoimage = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(original_canvas))
        masked_photoimage = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(masked_resized_2_canvas))

        return original_photoimage,masked_photoimage
    
    def set_label_images(self,canvas,original_image_canvas,masked_image_canvas):
        self.canvas = canvas
        self.original_image_canvas = original_image_canvas
        self.masked_image_canvas = masked_image_canvas 
        
    def refresh(self):
        original_photoimage,masked_photoimage = self.label_images()
        self.canvas.itemconfig(self.original_image_canvas,image = original_photoimage)
        self.canvas.itemconfig(self.masked_image_canvas, image = masked_photoimage)
        self.canvas.image1 = original_photoimage
        self.canvas.image2 = masked_photoimage 
        
    def run(self):
        self.load_image()
        self.current_mask_image, self.mask_img_fg = self.highlighter(self.multiple_masks, self.color_green)
        self.current_original_image = cv2.addWeighted(self.current_original_image, self.alpha, self.mask_img_fg, self.beta, self.gamma)
        
        # Create a resizable window with a larger default size
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)  
        cv2.resizeWindow('image', 1200, 800)  

        cv2.setMouseCallback('image', self.draw_circle)
        cv2.createTrackbar('Tolerance', 'image', self.tolerance, 5, self.change_tolerance)

        while True:
            # Apply zoom to the displayed image
            zoomed_image = self.apply_zoom(self.current_original_image)
            zoomed_mask = self.apply_zoom(self.current_mask_image)

            # Display the zoomed image and mask
            masked_and_original_image = cv2.hconcat([zoomed_image, zoomed_mask])
            cv2.imshow('image', masked_and_original_image)

            # Handle key events
            k = cv2.waitKey(1)

            # Close window if ESC is pressed or if the window is manually closed
            if k == 27 or cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        self.refresh()

    def apply_zoom(self, image):
        """Apply zoom to the image based on the current zoom level and center."""
        if self.zoom_level == 1.0:
            return image

        h, w = image.shape[:2]
        # Calculate the region of interest (ROI) for zooming
        new_w, new_h = int(w / self.zoom_level), int(h / self.zoom_level)
        x1 = max(0, self.zoom_x - new_w // 2)
        y1 = max(0, self.zoom_y - new_h // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)

        # Crop and resize the image to simulate zoom
        zoomed_image = image[y1:y2, x1:x2]
        zoomed_image = cv2.resize(zoomed_image, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed_image

    def draw_circle(self, event, x, y, flags, param):
        """Handle mouse events, including zooming and drawing."""
        
        # Convert zoomed coordinates to original image coordinates
        if self.zoom_level > 1.0:
            h, w = self.img_copied.shape[:2]  # Original image dimensions
            new_w, new_h = int(w / self.zoom_level), int(h / self.zoom_level)
            
            # Compute the top-left corner of the zoomed region
            x1 = max(0, self.zoom_x - new_w // 2)
            y1 = max(0, self.zoom_y - new_h // 2)

            # Convert clicked position (x, y) from zoomed view to original coordinates
            original_x = int(x1 + x / self.zoom_level)
            original_y = int(y1 + y / self.zoom_level)
        else:
            original_x, original_y = x, y  # No zoom, so use the raw coordinates

        if event == cv2.EVENT_MOUSEWHEEL:
            # Adjust zoom level based on scroll direction
            if flags > 0:  # Scroll up (zoom in)
                self.zoom_level *= 1.1
            else:  # Scroll down (zoom out)
                self.zoom_level /= 1.1
            self.zoom_level = max(1.0, min(self.zoom_level, 5.0))  # Limit zoom level
            self.zoom_x, self.zoom_y = x, y  # Set zoom center to current mouse position

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            list_hsv = self.hsv_finder(self.img_copied, original_y, original_x)
            self.hsv_(list_hsv)
            masks = self.hsv_ranging(self.listG)
            self.multiple_masks = self.hsv_comparing_with_new_mask(masks, self.multiple_masks)
            self.multiple_masks = self.hsv_comparing_within_the_list(self.multiple_masks)
            self.current_mask_image, self.mask_img_fg = self.highlighter(self.multiple_masks, self.color_green)
            self.current_original_image = cv2.addWeighted(self.img_copied, self.alpha, self.mask_img_fg, self.beta, self.gamma)
            self.append_to_pickle_file(self.multiple_masks)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.erase = True
            if self.multiple_masks:
                self.multiple_masks.pop()
            self.current_mask_image, self.mask_img_fg = self.highlighter(self.multiple_masks, self.color_green)
            self.current_original_image = cv2.addWeighted(self.img_copied, self.alpha, self.mask_img_fg, self.beta, self.gamma)

        elif event == cv2.EVENT_RBUTTONUP:
            self.erase = False


if __name__ == '__main__':
    imager = Imager(INPUT)

    some_img = cv2.resize(imager.current_mask_image, (255,255))
    h, w = some_img.shape[:2]

    root = tk.Tk()
    canvas = tk.Canvas(root,width=520, height=270)
    canvas.pack()
    icon = Image.open('./assets/highlighter_icon.png')
    photo = ImageTk.PhotoImage(icon)

    next_button = tk.Button(root, text=">>", command=imager.next_image, bg='red', fg='white')
    edit_button = tk.Button(root, image = photo, command=imager.run)
    prev_btn = tk.Button(root, text="<<", command=imager.previous_image, bg='red', fg='white')

    # refresh_btn.pack(padx=5, pady=15, side=tk.LEFT)
    next_button.pack(padx=5, pady=15, side=tk.RIGHT)
    edit_button.pack(padx=5, pady=15, side=tk.RIGHT)
    prev_btn.pack(padx=5, pady=15, side=tk.RIGHT)

    original_photoimage,masked_photoimage = imager.label_images()
    original_image_canvas = canvas.create_image(0, 0, image=original_photoimage, anchor=tk.NW)
    masked_image_canvas = canvas.create_image(w+10, 0, image=masked_photoimage, anchor=tk.NW)
    imager.set_label_images(canvas, original_image_canvas,masked_image_canvas)

    # Bind mouse scroll event for zoom functionality
    root.bind("<MouseWheel>", imager.apply_zoom)  # For zooming in/out on scroll

    root.mainloop()