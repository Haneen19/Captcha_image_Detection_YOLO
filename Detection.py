import logging
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

import cv2
import numpy as np
from ultralytics import YOLO
from utils import resource_path  # Assuming resource_path is defined in utils.py

# Paths
MODEL_PATH = resource_path("model/best.pt")
TARGET_SIZE = 640  # Target size for YOLO model (e.g., 640x640)
PADDING_COLOR = (0, 0, 0)  # Padding color (black)

# Initialize the YOLO model
model = YOLO(MODEL_PATH)

class CAPTCHAHandler:
    def __init__(self):
        self.driver = None
        self.is_running = False
        self.captcha_solution = ""
        self.labels = [
            'dog', 'person', 'cat', 'tv', 'car', 'meatballs', 'marinara sauce', 'tomato soup', 
            'chicken noodle soup', 'french onion soup', 'chicken breast', 'ribs', 'pulled pork', 
            'hamburger', 'cavity', 'L', 'y', '6', '8', '2', 'B', 'Y', 'H', 'R', 'Q', 't', '3', 'w', 
            'p', 'r', 'l', 'D', '1', 'd', 'b', 'v', 'M', 'E', 'X', '5', 'x', 'F', 'k', 'S', 'q', 'N', 
            'e', 'W', '4', 'h', 'f', 'J', "N'", 'g', 'n', '9', 'm', 'V', '7', '0', 's', 'T', 'a', 'z', 
            'Z', 'I', 'O', 'u', 'c', 'C', 'P', 'K', 'II', 'i', 'G', 'o', 'A', 'j', 'U'
        ]

    def setup_webdriver(self):
        """Setup and return configured Chrome WebDriver."""
        service = Service(resource_path("chromedriver/chromedriver.exe"))
        chrome_options = Options()
        chrome_options.add_argument("user-data-dir=C:/Users/Administrator/AppData/Local/Google/Chrome/User Data")
        chrome_options.add_argument("profile-directory=Default")
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--allow-insecure-localhost')
        return webdriver.Chrome(service=service, options=chrome_options)



    def start(self, update_status_callback):
        """Monitor dynamically for new tabs and handle the CAPTCHA page without pausing."""
        self.is_running = True
        try:
            # Initialize WebDriver if not already started
            if not self.driver:
                self.driver = self.setup_webdriver()
                update_status_callback("WebDriver initialized and browser started.")

            # Navigate to the main URL
            self.driver.get('https://www.purchasingprogramsaudi.com/#')
            update_status_callback("Navigated to the main URL. Waiting for new tabs to open.")
            print(f"Initial URL: {self.driver.current_url}")

            # Store the initial set of window handles
            initial_tabs = set(self.driver.window_handles)
            print(f"Initial tabs: {len(initial_tabs)}")

            # Main loop for monitoring new tabs
            while self.is_running:
                current_tabs = set(self.driver.window_handles)
                new_tabs = current_tabs - initial_tabs  # Detect new tabs that were not there before

                if new_tabs:
                    # Iterate over new tabs to detect URLs and handle them
                    for new_tab in new_tabs:
                        self.driver.switch_to.window(new_tab)  # Switch to the new tab
                        current_url = self.driver.current_url
                        print(f"New tab detected. Current URL: {current_url}")
                        update_status_callback(f"New tab opened with URL: {current_url}")

                        if current_url == "https://www.purchasingprogramsaudi.com/common/captcha.cfm":
                            print("CAPTCHA page detected.")
                            update_status_callback("CAPTCHA page detected. Handling CAPTCHA...")
                            self.handle_captcha(update_status_callback)  # Automate CAPTCHA handling
                            self.is_running = False  # Stop monitoring if CAPTCHA is handled
                            break  # Exit the loop if CAPTCHA is processed

                    # Update the initial tabs set to reflect the current state
                    initial_tabs = current_tabs

        except Exception as e:
            logging.error(f"Error in monitoring: {e}")
            update_status_callback(f"An error occurred: {e}")
        finally:
            self.stop()

    def stop(self):
        """Stop the CAPTCHA handling process."""
        self.is_running = False
        logging.info("Automation process stopped. Browser remains open.")
    def handle_captcha(self, update_status_callback):
        """Handle CAPTCHA detection and solving with continuous retries."""
        while True:
            try:
                if self.detect_captcha():
                    update_status_callback("CAPTCHA detected. Solving...")
                    success = self.solve_captcha()
                    
                    if success:
                        update_status_callback("CAPTCHA solved successfully!")
                        print("CAPTCHA solved successfully!")
                        return  # Exit if CAPTCHA is solved
                    
                    update_status_callback("Failed to solve CAPTCHA. Retrying...")
                else:
                    update_status_callback("CAPTCHA not detected. Retrying...")
            
            except Exception as e:
                logging.error(f"Error in handle_captcha method: {e}")
                update_status_callback(f"An error occurred: {e}")
                break  # Exit if an unexpected error occurs

    def detect_captcha(self):
        """Check if a CAPTCHA image is present."""
        try:
            self.driver.find_element(By.XPATH, "//img[contains(@src, 'captcha')]")
            logging.info("CAPTCHA detected.")
            return True
        except:
            logging.info("No CAPTCHA detected.")
            return False

    def solve_captcha(self):
        """Capture and solve CAPTCHA using YOLO model and retry until solved."""
        try:
            while True:  # Retry loop
                captcha_element = self.driver.find_element(By.XPATH, "//img[contains(@src, 'captcha')]")
                captcha_element.screenshot("captcha_image.png")
                logging.info("CAPTCHA image captured.")

                # Solve CAPTCHA using an external solution
                self.captcha_solution = self.extract_captcha_solution("captcha_image.png")
                logging.info(f"CAPTCHA solution: {self.captcha_solution}")

                # Input the solution and submit
                captcha_input = self.driver.find_element(By.ID, "captchaText")
                captcha_input.clear()
                captcha_input.send_keys(self.captcha_solution)
                self.driver.find_element(By.ID, "submit").click()

                # Check if still on CAPTCHA page
                if self.driver.current_url == "https://www.purchasingprogramsaudi.com/common/captcha.cfm":
                    logging.warning("CAPTCHA solve attempt failed. Retrying...")
                    continue  # Retry solving CAPTCHA
                else:
                    logging.info("CAPTCHA solved successfully, navigating away from CAPTCHA page.")
                    return True  # Exit retry loop if CAPTCHA is solved
        except Exception as e:
            logging.error(f"Error solving CAPTCHA: {e}")
            return False

    def extract_captcha_solution(self, image_path):
        """Extract CAPTCHA solution using YOLO."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            img_resized, scale_factors = self.resize_image(img, TARGET_SIZE, PADDING_COLOR)
            predictions = self.dynamic_prediction(img_resized)

            # Adjust predictions and reconstruct CAPTCHA solution
            adjusted_detections = self.adjust_coordinates(predictions, scale_factors)
            return self.arrange_and_label(adjusted_detections)
        except Exception as e:
            logging.error(f"Error extracting CAPTCHA solution: {e}")
            return ""

    def resize_image(self, image, target_size, padding_color):
        """Resize and pad image to match target size."""
        original_height, original_width = image.shape[:2]
        scale = min(target_size / original_width, target_size / original_height)
        new_width, new_height = int(original_width * scale), int(original_height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        padded_image = np.full((target_size, target_size, 3), padding_color, dtype=np.uint8)
        x_offset, y_offset = (target_size - new_width) // 2, (target_size - new_height) // 2
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

        scale_factors = {
            "scale": scale,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "original_size": (original_width, original_height),
        }
        return padded_image, scale_factors

    def dynamic_prediction(self, image):
        """Predict exactly 6 characters/letters with optimized detection parameters."""
        # Different confidence and IoU settings to target exactly 6 detections
        confidence_settings = [
            (0.3, 0.1),   # Lower confidence, lower IoU
            (0.4, 0.2),   # Moderate confidence, moderate IoU
            (0.5, 0.3),   # Higher confidence, higher IoU
        ]
        
        for conf, iou in confidence_settings:
            results = model(image, conf=conf, iou=iou)  # YOLOv8 'ultralytics' format
            boxes = results[0].boxes
            
            predictions = [
                [*box.xyxy[0].tolist(), box.conf[0].item(), int(box.cls[0].item())]
                for box in boxes
            ]
            
            # Return exactly 6 detections if found
            if len(predictions) == 6:
                return predictions
        
        # If no setting yields exactly 6 detections, return the closest match
        return predictions

    def adjust_coordinates(self, detections, scale_factors):
        """Adjust detection coordinates back to original image size."""
        adjusted_detections = []
        scale = scale_factors['scale']
        x_offset = scale_factors['x_offset']
        y_offset = scale_factors['y_offset']

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            adjusted_detections.append([
                (x1 - x_offset) / scale, 
                (y1 - y_offset) / scale, 
                (x2 - x_offset) / scale, 
                (y2 - y_offset) / scale, 
                conf, 
                cls
            ])
        return adjusted_detections

    def arrange_and_label(self, detections):
        """Arrange detections left-to-right and map to labels."""
        sorted_detections = sorted(detections, key=lambda x: x[0])  # Sort by x1 (left-to-right)
        return ''.join(self.labels[int(detection[5])] for detection in sorted_detections)

