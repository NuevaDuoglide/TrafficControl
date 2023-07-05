#include <Wire.h>
#include <Adafruit_VC0706.h>

// Initialize the camera object
Adafruit_VC0706 cam = Adafruit_VC0706(&Wire);

//Inizitaition of the LED pins
#define redLed
#define yellowLed
#define greenLed

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.println("Camera Test");

  //Inisiasi kamera
  if (cam.begin()) {
    Serial.println("Camera module found");
    cam.setImageSize(VC0706_640x480); // Set the image size (options: VC0706_640x480, VC0706_320x240, VC0706_160x120)
  } else {
    Serial.println("Camera module not found");
    while (1);
}

void loop() {
  // Capture a video frame
  if (cam.takePicture()) {
    // Get the image data
    uint8_t *loading;
    uint16_t len;

    if ((len = cam.frameLength()) > 0) {
      loading = cam.readPicture(len);

      // Send the image data to Raspberry Pi
      Serial.write(loading, len);
    }
  }
  //Lampu
  Serial.println("Data lampu");
  if (Serial.available()){
    command = Serial.readStringUntil('\n');
    command.trim();
    if (command.equals("red")){
      digitalWrite(redLed, HIGH);
      digitalWrite(yellowLed, LOW);
      digitalWrite(greenLed, LOW);
    }
    else if (command.equals("yellow")){
      digitalWrite(redLed, LOW);
      digitalWrite(yellowLed, HIGH);
      digitalWrite(greenLed, LOW);
    }
    else (command.equals("green")){
      digitalWrite(redLed, LOW);
      digitalWrite(yellowLed, LOW);
      digitalWrite(greenLed, HIGH);
  }
  delay (1000)
}
