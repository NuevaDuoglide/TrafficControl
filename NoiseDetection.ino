#include "arduinoFFT.h"

#define SAMPLES 1024             // SAMPLES-pt FFT. Must be a base 2 number. Max 128 for Arduino Uno.
#define SAMPLING_FREQUENCY 8192 // Ts = Based on Nyquist, must be 2 times the highest expected frequency.
#define MICROPHONE_PIN 12       // Pin connected to Max9814 Out

arduinoFFT FFT = arduinoFFT();

unsigned int samplingPeriod;
unsigned long microSeconds;

double vReal[SAMPLES]; // Create a vector of size SAMPLES to hold real values
double vImag[SAMPLES]; // Create a vector of size SAMPLES to hold imaginary values

void setup()
{
  Serial.begin(115200);                    // Baud rate for the Serial Monitor
  pinMode(MICROPHONE_PIN, INPUT);          // Set the microphone pin as input
  samplingPeriod = round(1000000 * (1.0 / SAMPLING_FREQUENCY)); // Period in microseconds
}

void loop()
{
  /* Sample SAMPLES times */
  for (int i = 0; i < SAMPLES; i++)
  {
    microSeconds = micros();    // Returns the number of microseconds since the Arduino board began running the current script.

    vReal[i] = analogRead(MICROPHONE_PIN); // Reads the value from the microphone pin, quantize it, and save it as a real term.
    vImag[i] = 0;                          // Makes the imaginary term 0 always

    /* Wait for the remaining time between samples if necessary */
    while (micros() < (microSeconds + samplingPeriod))
    {
      // Do nothing
    }
  }

  /* Perform FFT on samples */
  FFT.Windowing(vReal, SAMPLES, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.Compute(vReal, vImag, SAMPLES, FFT_FORWARD);
  FFT.ComplexToMagnitude(vReal, vImag, SAMPLES);

  /* Find peak frequency and print it */
  double peak = FFT.MajorPeak(vReal, SAMPLES, SAMPLING_FREQUENCY);
  Serial.print("Your Noise is ");
  Serial.print(peak);
  Serial.println(" Hz");

  delay(1000); // Wait for 1 second before detecting the frequency again
}
