import nidaqmx
import nidaqmx.system
from nidaqmx.constants import AcquisitionType
import numpy as np
from time import time, sleep
import matplotlib.pyplot as plt
import pickle
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import sounddevice as sd
from simulation.simulation import simulate_rx, simulate_tx


def turn_off_led():
    """
    write zero voltage to device
    :return:
    """
    with nidaqmx.Task() as tx:
        tx.ao_channels.add_ao_voltage_chan(device.name + "/ao0")
        tx.write(0, True)


def take_measurements(signal: np.ndarray, sampling_rate: float,offset=3.):
    """
    connect to device, send given signal, and take measurements
    :param signal:
    :param sampling_rate:
    :return: vals np.array of readed measurements
    """
    if device not in system.devices:
        vals = simulate_rx(simulate_tx(signal))
        sleep(signal.size / sampling_rate)
        return vals
    with nidaqmx.Task() as rx, nidaqmx.Task() as tx, nidaqmx.Task() as clock:

        # analog input channel
        rx.ai_channels.add_ai_current_chan(device.name + "/ai0")
        rx.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)

        # analog output channel
        tx.ao_channels.add_ao_voltage_chan(device.name + "/ao0")
        tx.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
        tx.stop() # make sure tx stopped
        tx.write(signal + offset) #write output signal

        #read data from rx
        tx.start()
        vals = np.array(rx.read(signal.size, 300))
        tx.stop()
    turn_off_led()
    return vals

def take_measurements2(duration: float, sampling_rate: float): #
    """
    take measurements only
    :param signal:
    :param sampling_rate:
    :return:
    """
    # if device not in system.devices:
    #     vals = simulate_rx(simulate_tx(signal))
    #     sleep(signal.size / sampling_rate)
    #     return vals
    with nidaqmx.Task() as rx:

        # analog input channel
        rx.ai_channels.add_ai_current_chan(device.name + "/ai0")
        # rx.ai_channels.add_ai_current_chan(device.name + "/ai7")
        rx.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)

        #read data
        vals = np.array(rx.read(round(duration*sampling_rate), 1.5*duration))
    return vals


def show_fft(array: np.ndarray, duration: int = 1):
    """
    shows the fft of given function
    :param array:
    :param duration:
    :return:
    """
    transform = fft(array)
    frequencies = fft.fftfreq(array.size, duration / array.size)
    plt.plot(frequencies[:array.size//2], np.abs(transform[:array.size//2]))
    plt.yscale("log")
    plt.show()
    return frequencies, transform




def distance(x1):
    """
    # todo: put in the notebook if I haven't yat
    :param x1:
    :return:
    """
    w = 570 #output frequency
    TimeInterval = 5 #time interval
    rate = 10000 #sampling rate
    x0 = 20 # initial location of diode on axis
    dis = x1 - x0 #
    offset = 3
    filename = f"measurments/distance/distance{dis}.pickle"  #output filename
    signal = np.sin(w * np.linspace(0, TimeInterval, TimeInterval * rate) * 2 * np.pi) / 2
    values = take_measurements(signal, rate, offset)

    plt.plot(np.arange(values.size)/rate, values)
    plt.xlabel("Sec")
    plt.ylabel("Amplitude")
    plt.show()
    plt.savefig(f"distance/measureDistance{dis}.jpg")
    #save data to file
    with open(filename, "wb+") as f:
        pickle.dump({"offset_voltage": offset, "signal": signal, "received": values, "interval": TimeInterval,
                     "rate": rate, "time": time(), "distance": dis, "filename": filename}, f)


def noise():
    """
    # todo: put in the notebook if I haven't yat
    :return:
    """
    w = 570
    duration = 5
    rate = 10000
    filename = f"noise_cover_photo_diode.pickle"
    sig = np.sin(w * np.linspace(0, duration, duration * rate) * 2 * np.pi) / 2
    values = take_measurements(sig, rate)
    plt.plot(values)
    plt.show()
    with open(filename, "wb+") as f:
        pickle.dump({"signal": sig, "received": values, "interval": duration, "rate": rate,
                     "time": time(), "fileName": filename}, f)


def song(filename, output_name="out.wav"):
    """
    # todo: put in the notebook if I haven't yat
    :param filename:
    :param output_name:
    :return:
    """
    sample_rate, data = wavfile.read(filename) #read song data
    data = np.array(data[:, 0] / (2 ** 15 - 1)) #normalize the data
    sliced_data = data[0:10 * sample_rate] #slice the data
    values = take_measurements(sliced_data, sample_rate) #transmit and receive
    values -= np.average(values) #avrage for move the offset
    values *= (2 ** 15 - 1) / max(values.max(), -values.min())
    values = values.round().astype("int16")
    wavfile.write(output_name, sample_rate, values)


def play_song(filename):
    """
    # todo: not need
    :param filename:
    :return:
    """
    sample_rate, data = wavfile.read(filename)
    chunk_size = sample_rate // 1000
    data = np.array(data[:, 0] / (2 ** 15 - 1))
    pos_range = iter(range(0, data.size, chunk_size))

    def callback(outdata: np.ndarray, frames: int, time, status) -> None:
        pos = next(pos_range)
        chunk = data[pos:pos + chunk_size+1]
        values = take_measurements(chunk, sample_rate * 2)
        values -= np.average(values)
        outdata[:, 0] = values

    with sd.OutputStream(sample_rate, chunk_size, channels=1, dtype="float32", callback=callback):
        sd.sleep(1000 * data.size // sample_rate)

def demodulate_am(signal: np.ndarray, rate: float, carrier_freq: float, band_radius: float = 40_000) -> np.ndarray:
    """
    demodulate amplitude modulation.
    :param signal: the modulated signal.
    :param rate: sample rate of the signal.
    :param carrier_freq: The frequency of the carrier.
    :param band_radius: The frequency limit of the modulated data.
    :return: the data of the signal.
    """
    t = np.linspace(0, signal.size / rate, signal.size)
    print(f"{len(t)=}")
    carrier = np.cos(t * carrier_freq * 2 * np.pi)
    return band_pass(signal * carrier, rate, 20, band_radius) * 4


def band_pass(signal: np.ndarray, rate: float, low: float, high: float):
    """
    filter out all the frequencies that outside the given band.
    :param signal: signal to filter.
    :param rate: the sample rate of the signal.
    :param low: the lower bound of the frequencies to pass.
    :param high: the upper bound of the frequencies to pass.
    :return: filtered signal.
    """
    sig_fft = fft.fft(signal)
    freq = fft.fftfreq(sig_fft.size, 1 / rate)
    abs_freq = np.abs(freq)

    sig_fft[abs_freq > high] = 0
    sig_fft[abs_freq < low] = 0

    return fft.ifft(sig_fft)

# finding the connections of instruments
system = nidaqmx.system.System.local()
for dev in system.devices:
    print(dev)

# select device
# device = system.devices["a"]
device = system.devices[input("write the device name to use: ")]
