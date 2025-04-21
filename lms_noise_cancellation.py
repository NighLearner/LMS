import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter, firwin

# Define parameters
fs = 8000  # Sampling frequency (Hz)
duration = 2  # Duration (seconds)
t = np.arange(0, duration, 1/fs)  # Time array

# Generate clean signal: 1000 Hz sine wave
clean_signal = np.sin(2 * np.pi * 1000 * t)

# Generate colored noise for effective LMS cancellation
cutoff = 500  # Low-pass filter cutoff (Hz)
nyq = 0.5 * fs
num_taps = 33
taps = firwin(num_taps, cutoff/nyq, window='hamming')
white_noise = np.random.normal(0, 1, len(t))
colored_noise = lfilter(taps, 1.0, white_noise)
colored_noise = colored_noise / np.std(colored_noise) * 0.1  # Scale to std 0.1

# Create noisy signal
noisy_signal = clean_signal + colored_noise

# LMS adaptive filter
M = 32  # Filter length
mu = 0.1  # Step size
N = len(t)
u = colored_noise  # Reference noise
d = noisy_signal  # Desired signal (noisy)
y = np.zeros(N)  # Filter output
e = np.zeros(N)  # Error signal (denoised)
w = np.zeros(M)  # Filter coefficients

for k in range(M-1, N):
    x_k = u[k - M + 1 : k + 1][::-1]  # Past M samples
    y[k] = np.dot(w, x_k)  # Filter output
    e[k] = d[k] - y[k]  # Error (denoised signal)
    w = w + mu * e[k] * x_k  # Update coefficients

# Plot signals
plt.figure(figsize=(12, 6))
plt.subplot(311)
plt.plot(t, clean_signal)
plt.title('Clean Signal')
plt.subplot(312)
plt.plot(t, noisy_signal)
plt.title('Noisy Signal')
plt.subplot(313)
plt.plot(t[M-1:], e[M-1:])
plt.title('Denoised Signal')
plt.tight_layout()
plt.savefig('lms_results.png')

# Save to WAV files
def float_to_int16(signal):
    signal = np.clip(signal, -1, 1)
    return np.int16(signal * 32767)

wavfile.write('clean_signal.wav', fs, float_to_int16(clean_signal))
wavfile.write('noisy_signal.wav', fs, float_to_int16(noisy_signal))
wavfile.write('denoised_signal.wav', fs, float_to_int16(e))