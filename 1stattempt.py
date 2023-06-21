import cv2
import numpy as np
from scipy.fftpack import fft
import RPi.GPIO as GPIO
import time
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# Traffic light pins
traffic_light_pins = [12, 25, 18]  # Green, Yellow, Red

# Function to detect car volume using OpenCV
def detect_car_volume(frame):
    # Convert the frame to grayscale for easier processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply image processing techniques (e.g., blurring, thresholding, etc.) to enhance car detection
    # You may need to experiment with different techniques and parameters based on your specific scenario

    # Example: Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Example: Apply adaptive thresholding to segment the image
    _, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Example: Perform morphological operations (e.g., erosion, dilation) to refine the binary image
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(threshold, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Example: Find contours of cars in the image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Example: Count the number of cars based on the detected contours
    car_count = len(contours)

    # Draw bounding boxes around the detected cars (optional)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with car count (optional)
    cv2.putText(frame, f"Car count: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Return the car count
    return car_count


# Function to detect ambulance using MAX 4466 sound sensor
def detect_ambulance():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(5, GPIO.IN)  # Connect the OUT pin of MAX 4466 to GPIO 15

    # Parameters for sound detection
    duration = 1  # Duration of sound detection in seconds
    fs = 44100  # Sampling frequency
    samples = int(fs * duration)
    threshold = 0.2  # Adjust this threshold based on your ambient noise level

    # Record audio
    audio = np.zeros(samples)
    for i in range(samples):
        audio[i] = GPIO.input(15)
        time.sleep(1.0 / fs)

    # Apply Fast Fourier Transform (FFT)
    fft_audio = fft(audio)

    # Calculate frequency
    frequencies = np.linspace(0.0, 1.0 / (2.0 * duration), samples // 2)
    amplitudes = 2.0 / samples * np.abs(fft_audio[0:samples // 2])

    # Find the maximum frequency
    max_frequency = frequencies[np.argmax(amplitudes)]

    # Check if the detected frequency is close to the ambulance siren frequency
    if abs(max_frequency - 550) < 20:
        return True
    else:
        return False


def fuzzy_logic_control(car_volume, ambulance_detected):
    # Define fuzzy sets and membership functions for car volume input variable
    car_volume_range = np.arange(0, 11, 1)
    green_time_range = np.arange(0, 11, 1)

    low_volume = fuzz.trimf(car_volume_range, [0, 0, 5])
    medium_volume = fuzz.trimf(car_volume_range, [3, 5, 8])
    high_volume = fuzz.trimf(car_volume_range, [6, 10, 10])

    # Define fuzzy sets and membership functions for green time output variable
    short_time = fuzz.trimf(green_time_range, [0, 0, 3])
    medium_time = fuzz.trimf(green_time_range, [2, 5, 8])
    long_time = fuzz.trimf(green_time_range, [6, 10, 10])

    # Create fuzzy variables
    car_volume_var = ctrl.Antecedent(car_volume_range, 'car_volume')
    ambulance_detected_var = ctrl.Antecedent([0, 1], 'ambulance_detected')
    green_time_var = ctrl.Consequent(green_time_range, 'green_time')

    # Associate membership functions with fuzzy variables
    car_volume_var['low'] = low_volume
    car_volume_var['medium'] = medium_volume
    car_volume_var['high'] = high_volume

    ambulance_detected_var.automf(2)  # Automatically create membership functions for ambulance_detected

    green_time_var['short'] = short_time
    green_time_var['medium'] = medium_time
    green_time_var['long'] = long_time

    # Define fuzzy rules
    rule1 = ctrl.Rule(car_volume_var['low'] & ambulance_detected_var['no'], green_time_var['long'])
    rule2 = ctrl.Rule(car_volume_var['low'] & ambulance_detected_var['yes'], green_time_var['short'])
    rule3 = ctrl.Rule(car_volume_var['medium'] & ambulance_detected_var['no'], green_time_var['medium'])
    rule4 = ctrl.Rule(car_volume_var['medium'] & ambulance_detected_var['yes'], green_time_var['short'])
    rule5 = ctrl.Rule(car_volume_var['high'] & ambulance_detected_var['no'], green_time_var['short'])
    rule6 = ctrl.Rule(car_volume_var['high'] & ambulance_detected_var['yes'], green_time_var['short'])

    # Create control system and simulate
    traffic_light_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
    traffic_light_sim = ctrl.ControlSystemSimulation(traffic_light_ctrl)

    # Set input values
    traffic_light_sim.input['car_volume'] = car_volume
    traffic_light_sim.input['ambulance_detected'] = ambulance_detected

    # Compute output values
    traffic_light_sim.compute()

    # Get the green time values for each traffic light
    green_time_values = {
        'traffic_light_1': traffic_light_sim.output['green_time'],
        'traffic_light_2': traffic_light_sim.output['green_time'],
        'traffic_light_3': traffic_light_sim.output['green_time'],
        'traffic_light_4': traffic_light_sim.output['green_time']
    }

    # Return the green time values for each traffic light
    return green_time_values

# Main program
def main():
    # Initialize GPIO pins for traffic light control
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(traffic_light_pins, GPIO.OUT)

    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Main loop
    while True:
        # Capture frame from the camera
        ret, frame = cap.read()

        # Process the frame to detect car volume
        car_volume = detect_car_volume(frame)

        # Check for ambulance
        ambulance_detected = detect_ambulance()

        # Apply fuzzy logic to determine traffic light timings
        green_time_values = fuzzy_logic_control(car_volume, ambulance_detected)

        # Control the traffic lights based on the determined timings
        for pin, green_time in zip(traffic_light_pins, green_time_values.values()):
            GPIO.output(pin, GPIO.HIGH)  # Turn on the traffic light
            time.sleep(green_time)  # Wait for the specified green time
            GPIO.output(pin, GPIO.LOW)  # Turn off the traffic light

        # Display the processed frame with car count and other information if required
        cv2.imshow("Frame", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
