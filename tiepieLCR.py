"""
.. module:: TiePieLCR
   :synopsis: This class can be used to get demodulated data from a single TiePieLCR
.. moduleauthor:: Martijn Schouten <github.com/martijnschouten>
"""

from __future__ import annotations
from libtiepie.generator import Generator
import sys
import libtiepie
import time
import numpy as np
from scipy import signal
import gc
try:
    import torch as tr
    test = tr.zeros(3)
    if tr.cuda.is_available():
        gpu_available = True
    else:
        gpu_available = False
except Exception as e:
    print(e)
    gpu_available = False
import matplotlib.pyplot as plt
from TiePieLCR_settings import TiePieLCR_settings
from libtiepie import exceptions


class TiePieLCR:
    
    I2C_ADDRESS = int('0b01100000',2)
    """I\ :superscript:`2`\ C address of the TLC59116 chip in the TiePieLCR analog frontend"""

    I2C_RANGE1 = int('0b00000001',2)
    """Mask for enabling the first (370uA/V) transimpedance amplifier"""

    I2C_RANGE2 = int('0b00000100',2)
    """Mask for enabling the second (5uA/V) transimpedance amplifier"""

    I2C_RANGE3 = int('0b00010000',2)
    """Mask for enabling the third (390pC/V) transimpedance/charge amplifier"""

    I2C_RANGE4 = int('0b01000000',2)
    """Mask for enabling the third (3.9pC/V) transimpedance/charge amplifier"""

    I2C_GAIN = int('0b01000001',2)
    """Mask for enabling the 50 times gain amplifier"""

    I2C_CARSELECT = int('0b00010000',2)
    """Mask for selecting the signal coming directly from the function gegenerator as a reference instead a transimpedance/charge amplifier"""

    I2C_NONE  = int('0b00000000',2)
    """Mask for selecting 1x gain or none of the transimpedance amplifiers"""

    I2C_LEDOUT0 = int('0x14',16)
    """Register that controls which of the four transimpedance/charge amplifiers is used"""

    I2C_LEDOUT1 = int('0x16',16)
    """Register that controls which the gain and if the carrier used as a reference"""

    I2C_MODE1 = int('0x00',16)
    """Register that is written 0, proably to put the chip in normal mode (needs verification)"""


    def __init__(self, all_settings,instance):
        """Code run when the TiePieLCR object is initialised. During initialisation the object is filled with one of the settings in the all_settings list. Which one is loaded is determined by the instance parameter and lockins determines the total number of lockins that are availabe.

        :param all_settings: A list of the TiePieLCR_settings objects that contain the settings for this lockin
        :param instance: The index in the all_settings list that belongs to this tiepieLCR's settings
        :return: None
        :rtype: None
        """

        self.lockins = 0
        """Total number of lockins that is being used, corresponds to the number of instances of the TiePieLCR class"""

        self.block_number = 0
        """Number of blocks of data that have been retrieved from the TiePie oscilloscopes since the start of this measurement"""

        self.block_number_process =  0
        """Number of blocks of data that have been processed"""
                    
        self.state = 0
        """State the system. The state is only set for the tiepieLCR instance that is used for getting the data (often called masterLCR)

        * state 0: Not initialised
        * state 1: Initialising
        * state 2: Running
        * state 3: Stopping
        * state 4: Updating

        """

        self.old_signal_data = []
        """The previous package of the signal data. This data will be needed to get rid of the edge effects. Master tiepieLCR only."""

        self.old_reference_data = []
        """The previous package of reference data. This data will be needed to get rid of the edge effects. Master tiepieLCR only."""
        
        self.old_time = None
        """The timestamps belonging to previous package of reference data. Master tiepieLCR only."""
        
        self.gen = None
        """A list of instances of the libtiepie Generator object. One for each tiepie. Master tiepieLCR only."""

        self.i2c = []
        """A list of instances of the libtiepie I\ :superscript:`2`\ C  objects. These are used to control the I2C bus that goes from the back of the tiepie to the TLC59116 chip in the TiePieLCR analog frontend. Master tiepieLCR only."""

        self.scp = None
        """The libtiepie oscilloscope object. Used to get data from the TiePie HS5's"""

        self.settings = TiePieLCR_settings()
        """An instance of the TiePieLCR class that will contain the settings of this TiePieLCR"""

        self.clipping_counter_ref = False
        """How many blocks the reference should not be clipping before it will be considered good (and will become green). Master tiepieLCR only."""

        self.clipping_counter_sig = False
        """How many blocks the signal should not be clipping before it will be considered good (and will become green). Master tiepieLCR only."""

        self.base_time_vector = None
        """A vector with the time at which each sample in a block of data is taken, relative to the start of the block.  Master tiepieLCR only."""

        self.base_freq_vector = None
        """A vector with the frequency of each sample in output of the fft.  Master tiepieLCR only."""

        self.window_cpu = None
        """The window that will be applied to a block of data before applying the fft and is located in the CPU memory. Master tiepieLCR only."""

        self.window_gpu = None
        """The window that will be applied to a block of data before applying the fft and is located in the GPU memory. Master tiepieLCR only."""

        self.offset_win = None 
        """To calculate the offset the inverse fourier transform of the lowest frequent fourier coefficients is taken. However because a window was used the inverse window needs to be applied. This vector will contain the inverse fourier transform of lowest fourier coeffients of the window, which can be used to apply the inverse window. Master tiepieLCR only."""

        self.sos_offset = None
        """The lowpass filter coefficients that will be applied to the offset. Master tiepieLCR only."""

        self.filt_win = None
        """To calculate band pass filtered signals during the low frequency algorithm the inverse fourier transform of relevant fourier coefficients is taken. However because a window was used the inverse window needs to be applied to reduce noise. This vector will contain the inverse fourier transform of the fourier coeffients window used for this frequency, which can be used to apply the inverse window. Master tiepieLCR only."""

        self.b_bandpass = None
        """The fir bandpass filter coefficients that will be used during the low frequent demodulation algorithm. Master tiepieLCR only."""

        self.sos_decimate = None
        """The filter coefficients that will be used for the final low pass filter of both algorithms. Master tiepieLCR only."""

        self.demodulate_save_buffer = None
        """"Buffer that will store the result of the demodulation. The size of this buffer will be determined by :attr:`TiePieLCR_settings.save_memory` """

        self.ref_offset_save_buffer = None
        """"Buffer that will store the result of the offset of the reference. The size of this buffer will be determined by :attr:`TiePieLCR_settings.save_memory`"""

        self.sig_offset_save_buffer = None
        """"Buffer that will store the result of the offset of the signal. The size of this buffer will be determined by :attr:`TiePieLCR_settings.save_memory`"""

        self.timestamps = None
        """"Contains a vector with estimated timestamps for each demodulation"""

        self.zi_ref_offset_downsample = np.zeros((1, 2))
        """State of the final low pass filter that is being applied to the offset of the reference"""

        self.zi_sig_offset_downsample = np.zeros((1, 2))
        """State of the final low pass filter that is being applied to the offset of the signal"""

        self.zi_buf1_R = []
        """State of the final low pass filter that is being applied to the real part of the output of demodulation alogrithm. For the high frequency algorithm this is the impedance, for the low frequency algorithm this is the reference times the signal"""
        
        self.zi_buf1_I = []
        """State of the final low pass filter that is being applied to the imaginary part of the output of demodulation alogrithm. For the high frequency algorithm this is the impedance, for the low frequency algorithm this is the reference times the signal"""
        
        self.zi_buf2_R = []
        """State of the final low pass filter that is being applied to the real part of the output of demodulation alogrithm. This filter is only applied in case low frequency algorithm is used and then is applied to the square of the reference"""
        
        self.zi_buf1_I = []
        """State of the final low pass filter that is being applied to the imaginary part of the output of demodulation alogrithm. This filter is only applied in case low frequency algorithm is used and then is applied to the square of the reference"""
        
        self.zi_bandpass_ref = []
        """State of the bandpass filter used in the low frequency algorithm to filter out specific frequencies from the reference"""

        self.zi_bandpass_sig = []
        """State of the bandpass filter used in the low frequency algorithm to filter out specific frequencies from the signal"""
        
        self.zi_hilbert = []
        """State of the bandpass filter used in the low frequency algorithm to store the state of the function determining the Hilbert transform of the reference"""

        self.set_settings(all_settings,instance,self)
        self.update_base_vectors()
        
    def update_base_vectors(self):
        """(re)Initialise some vectors that are needed for the demodulation algorithm, but depend on the settings.

        :return: None
        :rtype: None
        """
        print("updated base vectors")
        self.base_time_vector = np.arange(self.settings.get_block_size()*2)/self.settings.get_sample_frequency()
        self.base_freq_vector = np.arange(self.settings.get_sub_block_size())/self.settings.get_sub_block_size()/2*self.settings.get_sample_frequency()
        self.window_cpu = signal.blackmanharris(2*self.settings.get_sub_block_size())
        if gpu_available:
            device = tr.device("cuda:0")
            window_torch = tr.from_numpy(self.window_cpu)
            self.window_gpu = window_torch.to(device)
        
        self.update_filter_coefs()

    def update_filter_coefs(self):
        """(re)Initialise the coefficients of the band and low pass filters as well as the processed window functions that are used for the inverse window

        :return: None
        :rtype: None
        """
        if self.window_cpu is None:
            self.update_base_vectors()
          
        print("updating filter coefs")
        
        #calculate the fft of the used window
        window_fft = np.fft.rfft(self.window_cpu)  

        #calculate offset_win 
        use = range(0,self.settings.get_offset_sub_block_size()+1)
        self.offset_win = np.fft.irfft(window_fft[use])*(self.settings.get_offset_sub_block_size())/self.settings.get_sub_block_size() #needs to be multiplied with offset_sub_block_size instead of offset_sub_block_size+1 for amplitude to be correct

        #calculate lowpass filter coefficients that will be applied to the offset
        fs = self.settings.get_offset_sample_frequency()
        Wn = self.settings.offset_bandwidth/fs*2
        self.sos_offset = signal.butter(2,Wn, analog=False, btype='low', output='sos')

        #calculate the filter coefficients for the demodulation algorithms
        length = int(self.settings.get_output_sub_block_size()/2)
        self.filt_win = []
        self.b_bandpass = []
        self.sos_decimate = []
        for i1 in range(len(self.settings.dem_freqs)):
            
            fs = self.settings.get_output_sample_freq()
            Wn = self.settings.bandwidth/fs*2
            self.sos_decimate.append(signal.butter(2,Wn, analog=False, btype='low', output='sos'))

            #frequency bin below the modulation frequency
            n_freq_min = int(np.floor(self.settings.dem_freqs[i1]/self.settings.get_df()-0.000001))
            
            if n_freq_min > length + 1 +self.settings.neglect_bins:
                #use the high frequency algorithm 
                self.filt_win.append([])
                self.b_bandpass.append([])
            else:
                #use the low frequency algorithm 
                n_len = length*self.settings.get_output_oversample_ratio()*2
                use = np.arange(n_len+1)
                win_windowed = window_fft[use]

                self.filt_win.append(np.fft.irfft(win_windowed))
                self.filt_win[i1] = self.filt_win[i1] * n_len/self.settings.get_sub_block_size() 

                fnyq = n_len*self.settings.get_sub_block_freq()/2
                Wn = self.settings.dem_freqs[i1]/fnyq
                width = self.settings.get_demodulation_bandwidth()/fnyq
                self.b_bandpass.append(self.design_bandpass_fir_filter(self.settings.get_sub_block_size(),Wn,width))



    def set_settings(self,all_settings:list[TiePieLCR_settings],instance,master_lcr: TiePieLCR):
        """(re)Initialise the coefficients of the band and low pass filters as well as the processed window functions that are used for the inverse window

        :param all_settings: A list of the TiePieLCR_settings objects that contain the settings for this lockin
        :param instance: The index in the all_settings list that belongs to this tiepieLCR's settings
        :param master_lcr: The TiePieLCR object that will be used get the data
        :return: True if succesfull, False if not
        :rtype: Boolean
        """
        # print("Updating settings. Instance: %i"%(instance))
        # print("Following flags are set:")
        # print("restart_required = %r"%(all_settings[instance].restart_required))
        # print("reference_update_required = %r"%(all_settings[instance].reference_update_required))
        # print("multisine_update_required = %r"%(all_settings[instance].multisine_update_required))
        # print("gen_restart_required = %r"%(all_settings[instance].gen_restart_required))
        # print("base_vector_update_required = %r"%(all_settings[instance].base_vector_update_required))
        # print("gen_offset_update_required = %r"%(all_settings[instance].gen_offset_update_required))
        # print("gen_amplitude_update_required = %r"%(all_settings[instance].gen_amplitude_update_required))
        # print("dem_freq_update_required = %r"%(all_settings[instance].dem_freq_update_required))
        old_state = master_lcr.state
        if master_lcr.state == 2:
            master_lcr.state = 4

        self.settings.load_calibration()
        self.settings = all_settings[instance]
        self.lockins = len(all_settings)
        self.inst = instance
        
        if self.settings.base_vector_update_required:
            self.update_base_vectors()
        if self.settings.dem_freq_update_required:
            #self.reset_buffers()
            self.update_filter_coefs()
        if self.settings.reference_update_required:
            if not master_lcr.select_reference(self.inst,self.settings.reference_setting,self.settings.gain_setting):
                return False
        if self.settings.gain_update_required:
            if not master_lcr.select_LCR_gain(self.inst,self.settings.reference_setting, self.settings.gain_setting):
                return False
        if self.settings.gen_offset_update_required:
            if master_lcr.gen:
                if master_lcr.state == 4:
                    master_lcr.scp.stop()
                    print('restarting scope because gen_offset_update_required')
                master_lcr.gen[self.inst].offset = self.settings.get_gen_offset()
                if master_lcr.state == 4:
                    master_lcr.scp.start()
        if self.settings.gen_amplitude_update_required:
            if master_lcr.gen:
                if master_lcr.state == 4:
                    master_lcr.scp.stop()
                    print('restarting scope because gen_amplitude_update_required')
                master_lcr.gen[self.inst].amplitude = self.settings.get_gen_amplitude()
                if master_lcr.state == 4:
                    master_lcr.scp.start()
        if self.settings.multisine_update_required and not self.settings.gen_restart_required:
            #self.set_multisine()
            if master_lcr.state == 4:
                master_lcr.scp.stop()
                print('restarting scope because multisine_update_required')
                master_lcr.gen[self.inst].set_data(self.settings.get_multisine_vector())
                master_lcr.gen[self.inst].frequency = self.settings.get_gen_sample_freq()
                master_lcr.scp.start()
        if self.settings.restart_required and not self.settings.gen_restart_required:
            if master_lcr.state == 4:
                master_lcr.stop_stream()
                print('restarting scope because restart_required')
                master_lcr.start_stream(all_settings)
        if self.settings.gen_restart_required:
            if master_lcr.state == 4:
                master_lcr.stop_stream()
                master_lcr.stop_gen(self.inst,False)
                print('restarting scope and gen because gen_restart_required')
                master_lcr.start_awg(self.inst,self.settings)
                master_lcr.start_stream(all_settings)            
        

        master_lcr.state = old_state        
        return True

        

    def open_scope(self):
        """Initialise the connection to the oscillscopes inside the TiePie's

        :return: Empty dict when succesfull, dict with 'error' containing a string that describes the error.
        :rtype: Dict
        """
        
        package = {}
        print("updating device list")
        libtiepie.device_list.update()
        
        # Try to open an oscilloscope with stream measurement support:
        self.scp = None
        for item in libtiepie.device_list:
            if item.can_open(libtiepie.DEVICETYPE_OSCILLOSCOPE):
                if item.types == 1:
                    self.settings.set_serial_numbers(np.array(item.contained_serial_numbers))
                elif item.types == 7:
                    self.settings.set_serial_numbers(np.array([item.serial_number]))
                else:
                    print("unkown device type")
                self.scp = item.open_oscilloscope()
                
                if self.scp.measure_modes & libtiepie.MM_STREAM:
                    break
                else:
                    self.scp = None
                
        if self.scp:
            if self.lockins*2 > len(self.scp.channels):
                print("Not enough oscilloscopes found. Check the connections and the number of lockins you have requested")
                package['error'] = b"Not enough oscilloscopes found. Check the connections and the number of lockins you have requested"
                return package
            else:
                return package
        else:
            package['error'] = b"Failed to open scope. Is the handyscope plugged in? Is it open in another program? Have you tried restartig the app?"
            return package
    
    def open_gen(self):
        """Initialise the connection to arbitraty waveform generators in the TiePie's

        :return: True if succesfull, False if not
        :rtype: Boolean
        """
        if not self.scp:
            raise Exception('The scope should be opened first')

        print("updating device list")
        libtiepie.device_list.update()
        self.gen = [Generator]*self.settings.get_serial_numbers().size
        for item in libtiepie.device_list:
            if item.can_open(libtiepie.DEVICETYPE_GENERATOR):
                serial_number = item._get_serial_number()
                loc = np.where(self.settings.get_serial_numbers() == serial_number)
                    
                temp_gen = item.open_generator()
                if temp_gen.signal_types & libtiepie.ST_ARBITRARY:
                    self.gen[loc[0][0]] = temp_gen
                print('generator serial number: '+str(item._get_serial_number()))

        if len(self.gen) > 0:
            return True
        else:
            print('No generator available with arbitrary support!')
            return False

    def __open_i2c(self):
        """Initialise the connection to the \ :superscript:`2`\ busses on the back of the TiePie's

        :return: True if succesfull, False if not
        :rtype: Boolean
        """

        # Search for devices:
        libtiepie.device_list.update()

        # Try to open an I2C host:
        self.i2c = []
        serial_numbers = []
        for item in libtiepie.device_list: 
            if item.can_open(libtiepie.DEVICETYPE_I2CHOST):
                serial_numbers.append(item.serial_number)
                self.i2c.append(item.open_i2c_host())
                print('i2c serial number: '+str(item._get_serial_number()))

        indexes = np.argsort(serial_numbers)
        
        i2c_old = self.i2c
        self.i2c = []
        for i1 in range(len(i2c_old)):
            self.i2c.append(i2c_old[int(indexes[i1])])

        if len(self.i2c)>0:
            try:
                for i1 in range(len(self.i2c)):
                    self.i2c[i1]._set_speed(400e3)
            except Exception as e:
                print("Can't connect to TiePieLCR. Make sure it's on and connected")
                print(e)
                return False
            return True
        else:
            print('No i2c port found')
            return False


    def select_LCR_gain(self,instance,reference_setting,gain_setting):
        """Set's the gain of the instrumentation amplifier in the TiePieLCR analog frontend.

        :param instance: The index of the tiepie connected to the frontend of which the gain should be set
        :param reference_setting: Which reference should be selected. This value is needed because it might be set by the same register.
        :param gain_setting: If 1 the gain of the instrumentation amplifer will be 50, otherwise it will be 1
        
        :return: True if succesfull, False if not
        :rtype: Boolean
        """

        print("selecting gain of %s with reference is %s and gain is %s"%(instance,reference_setting,gain_setting))

        if self.settings.no_frontend:
            return False

        if len(self.i2c) == 0:
            if not self.__open_i2c():
                print("couldn't open i2c")
                return False

        try:
            if gain_setting == 1:
                if reference_setting == 5 or reference_setting == 6:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_GAIN|self.I2C_CARSELECT)
                else:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_GAIN)
            elif gain_setting == 0:
                if reference_setting == 5 or reference_setting == 6:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_CARSELECT)
                else:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_NONE)
            else:
                print('Unkown gain requested')
                return False
        except Exception as e:
            print('exception while setting gain')
            print(e)
            return False
        return True

    def select_reference(self,instance,reference_setting,gain_setting):
        """Set's which signal is used as the reference. This can either be the output of one of the transimpedance amplifiers or the output of the generator. The available references are:

        * 0: No reference selected
        * 1: First transimpedance amplifier
        * 2: Second transimpedance amplifier
        * 3: Third transimpedance amplifier
        * 4: Fourth transimpedance amplifier
        * 5 and 6: The output of the generator

        :param instance: The index of the tiepie connected to the frontend of which the gain should be set
        :param reference_setting: Which reference should be selected.
        :param gain_setting: If 1 the gain of the instrumentation amplifer will be 50, otherwise it will be 1. This value is needed because it might be set by the same register.
        
        :return: True if succesfull, False if not
        :rtype: Boolean
        """
        print("selecting reference of %s with reference is %s and gain is %s"%(instance,reference_setting,gain_setting))

        if self.settings.no_frontend:
            return False

        if len(self.i2c) == 0:
            if not self.__open_i2c():
                print("couldn't open i2c")
                return False
        try:
            if reference_setting== 0:
                self.__write_register(instance,self.I2C_LEDOUT0,self.I2C_NONE)
                time.sleep(0.001)
                self.__write_register(instance,self.I2C_MODE1,0)
                if gain_setting == 1:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_GAIN)
                else:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_NONE)
            elif reference_setting == 1:
                self.__write_register(instance,self.I2C_LEDOUT0,self.I2C_RANGE1)
                time.sleep(0.001)
                self.__write_register(instance,self.I2C_MODE1,0)
                if gain_setting == 1:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_GAIN)
                else:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_NONE)
            elif reference_setting == 2:
                self.__write_register(instance,self.I2C_LEDOUT0,self.I2C_RANGE2)
                time.sleep(0.001)
                self.__write_register(instance,self.I2C_MODE1,0)
                if gain_setting == 1:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_GAIN)
                else:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_NONE)
            elif reference_setting == 3:
                self.__write_register(instance,self.I2C_LEDOUT0,self.I2C_RANGE3)
                time.sleep(0.001)
                self.__write_register(instance,self.I2C_MODE1,0)
                if gain_setting == 1:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_GAIN)
                else:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_NONE)
            elif reference_setting == 4:
                self.__write_register(instance,self.I2C_LEDOUT0,self.I2C_RANGE4)
                time.sleep(0.001)
                self.__write_register(instance,self.I2C_MODE1,0)
                if gain_setting == 1:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_GAIN)
                else:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_NONE)
            elif reference_setting == 5 or reference_setting == 6:
                self.__write_register(instance,self.I2C_LEDOUT0,self.I2C_NONE)
                time.sleep(0.001)
                self.__write_register(instance,self.I2C_MODE1,0)
                if gain_setting == 1:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_GAIN|self.I2C_CARSELECT)
                else:
                    self.__write_register(instance,self.I2C_LEDOUT1,self.I2C_CARSELECT)
            else:
                print('Unkown reference channel selected')
                return False
        except Exception as e:
            print('exception while setting reference '+str(reference_setting))
            print(e)
            return False       
        return True
    
    def __write_register(self,instance,register,value):
        """Write a value to a register in the TLC59116 chip

        :param instance: The index of the tiepie connected to the frontend that contains the TLC59116 of which a register should be written.
        :return: None
        :rtype: None
        """
        self.i2c[instance].write_byte_byte(self.I2C_ADDRESS, register, value)

    def __read_register(self,instance,register):
        """Read a value from a register in the TLC59116 chip

        :param instance: The index of the tiepie connected to the frontend that contains the TLC59116 of which a register should be read.
        :return: The read value
        :rtype: int
        """
        self.i2c[instance].write_byte(self.I2C_ADDRESS,register)
        return self.i2c[instance].read_byte(self.I2C_ADDRESS)

    def start_awg(self,instance,settings):
        """Configure and start a arbitrary waveform genenerator in one of the TiePie's

        :param instance: The index of the tiepie of which the arbitrary waveform genenerator should be set
        :return: True if succesfull, False if not
        :rtype: Boolean
        """
        data = settings.get_multisine_vector()

        if not self.gen:
            print('No generator has been opened yet')
            return False
        
        # Set signal type:
        self.gen[instance].signal_type = libtiepie.ST_ARBITRARY

        # Select frequency mode:
        self.gen[instance].frequency_mode = libtiepie.FM_SAMPLEFREQUENCY

        # Set sample frequency:
        self.gen[instance].frequency = settings.get_gen_sample_freq()

        # Set amplitude:
        self.gen[instance].amplitude = settings.get_gen_amplitude()

        # Set offset:
        self.gen[instance].offset = settings.get_gen_offset()

        # Enable output:
        self.gen[instance].output_on = True

        if data is None:
            print('awg signal not yet set')
            return False
        else: 
            try:
                self.gen[instance].set_data(data)
            except Exception as e:
                print('Could not set generator data: ' + str(e))
                return False

        # Start signal generation:
        self.gen[instance].start()
        return True


    def start_stream(self,all_settings):
        """Configure and start streaming data from TiePie HS5's

        :param all_settings: A list of the TiePieLCR_settings objects that contain the settings for each lockin
        :return: True if succesfull, False if not
        :rtype: Boolean
        """
        if not self.scp:
            print('No oscilloscope has been opened yet')
            return False

        self.state = 1
        # Set measure mode:
        try:
            self.scp.measure_mode = libtiepie.MM_STREAM
        except Exception as e:
            print(e)
            return False
        # Set sample frequency:
        self.scp.sample_frequency = self.settings.get_sample_frequency()

        # Set record length:
        self.scp.record_length = int(self.settings.get_block_size())

        self.scp.resolution = 16

        # For all channels:
        for i2 in range(self.lockins):
            for i1 in range(2):
                #chn = i1*self.lockins+i2
                chn = i1+i2*2
                # Enable channel to measure it:
                self.scp.channels[chn].enabled = all_settings[i2].enabled[i1]
                
                # Set range:
                self.scp.channels[chn].range = all_settings[i2].scope_range_list[all_settings[i2].scope_ranges[i1]] 

                # Set coupling:
                self.scp.channels[chn].coupling = int(all_settings[i2].scope_couplings_list[all_settings[i2].scope_couplings[i1]])

        self.reset_buffers()

        # Start measurement:
        try:
            self.scp.start()
        except Exception as e:
            print(e)
            return False

        self.state = 2
        return True

    def reset_buffers(self):
        """Reset variables used by the demodulation algorithm 

        :return: None
        :rtype: None
        """
        
        print("reset buffers")
        self.demodulate_save_buffer = np.zeros((int(self.settings.save_memory),self.settings.get_number_of_demodulate_freqs()),dtype=np.cdouble)
        self.ref_offset_save_buffer = np.zeros(int(self.settings.offset_save_memory),dtype=np.cdouble)
        self.sig_offset_save_buffer = np.zeros(int(self.settings.offset_save_memory),dtype=np.cdouble)
        self.timestamps = np.zeros(int(self.settings.save_memory),dtype=np.cdouble)

        self.block_number = 0
        self.block_number_process=  0
        self.old_signal_data = []
        self.old_reference_data = []
        self.old_time = None
        self.zi_ref_offset_downsample = np.zeros((1, 2))
        self.zi_sig_offset_downsample = np.zeros((1, 2))
        self.zi_buf1_R = []
        self.zi_buf1_I = []
        self.zi_buf2_R = []
        self.zi_buf2_I = []
        self.zi_bandpass_ref = []
        self.zi_bandpass_sig = []
        self.zi_hilbert = []
        for i1 in range(self.settings.get_number_of_demodulation_freqs()):
            self.zi_buf1_R.append(np.zeros((1, 2)))
            self.zi_buf1_I.append(np.zeros((1, 2)))
            self.zi_buf2_R.append(np.zeros((1, 2)))
            self.zi_buf2_I.append(np.zeros((1, 2)))
            self.zi_bandpass_ref.append([])
            self.zi_bandpass_sig.append([])
            self.zi_hilbert.append([])


    def get_data(self,all_settings):
        """Get data from oscilloscope in the TiePie HS5's. Return a dict with obtained data. This dict contains the following variables:

        * 'reference': A 2D numpy array with the voltage measured on the reference channels. The first index is the lockin the reference belongs to, the second the sample number.
        * 'signal': A 2D numpy array with the voltage measured on the signal channels. The first index is the lockin the reference belongs to, the second the sample number.
        * 'time': A 1D numpy vector with the time since the start of the measurement for each sample in the 'reference' and 'signal'
        
        In case of an error the dict will not contain the above keys but instead will contain one of the following:
        
        * 'gone': This indicates the connection to HS5 was unsuccesfull of no object could be found. This usually means the object has started up completely.
        * 'Error': Another error occured while getting data.
        * 'Warning': The TiePie's could keep up the data flow and an overflow occured. Could be solved by reducing sampling rate or choosing USB ports which are on seperate controllers.

        The function always returns one sub-block data from the previous block. This is because the demodulation algorithms only keep the center part of two sub-blocks.

        :return: A package with the obtained data
        :rtype: Dict
        """
        get_more_data = True
        while(get_more_data):
            package = {}
            
            try:
                watchdog = 0
                while not (self.scp.is_data_ready or self.scp.is_data_overflow):
                    time.sleep(0.001)
                    watchdog = watchdog + 1
                    if watchdog > 100:
                        package['timeout'] = b'Data timeout'
                        self.state = 0
                        return package

            except (exceptions.ObjectGoneError,exceptions.UnsuccessfulError):
                print('Gone or uncuccesfull!')
                package['gone'] = b'warning: You were too damn fast, give me some time to start up and try again'
                self.state = 0
                return package
            except:
                print('Error!')
                package['error'] = b'error: could not determine if data is ready or if there is an overflow'
                return package
            
            if self.scp.is_data_overflow:
                print('Data overflow!')
                package['warning'] = b'error: data overflow. Try reducing data rate, increasing sub plots or a plugin in your notebook power supply'
                self.scp.stop()
                self.scp.start()
                return package
            
            # Get data:
            try:
                data_raw = self.scp.get_data()
            except:
                print('error while getting data: ' + str(sys.exc_info()[0]))
                package['error'] = b'error: while getting data: ' + bytes(str(sys.exc_info()[0]),'utf-8')
                return package            
            
            if self.state != 2:
                package['warning'] = b'warning: incorrect state'
                print('warning: incorrect state')
                return package
            
            sub_blocks = self.settings.get_sub_blocks()
            sub_block_size = self.settings.get_sub_block_size()

            new_reference_data = np.zeros((self.lockins,sub_blocks*sub_block_size))
            new_signal_data = np.zeros((self.lockins,sub_blocks*sub_block_size))
            reference_data = np.zeros((self.lockins,(sub_blocks+1)*sub_block_size))
            signal_data = np.zeros((self.lockins,(sub_blocks+1)*sub_block_size))
            for i1 in range(self.lockins):
                ref_data = np.frombuffer(data_raw[i1*2],dtype='f')
                sig_data = np.frombuffer(data_raw[i1*2+1],dtype='f')
                
                new_reference_data[i1,:] = ref_data*float(all_settings[i1].get_reference_gain())
                new_signal_data[i1,:] = sig_data/float(all_settings[i1].get_gain_value())
                
                if len(self.old_signal_data) > 0:
                    reference_data[i1,:] = np.concatenate((self.old_reference_data[i1,:],new_reference_data[i1,:]))
                    signal_data[i1,:] = np.concatenate((self.old_signal_data[i1,:],new_signal_data[i1,:]))
                    
                    get_more_data = False
                
            
            self.old_reference_data = np.zeros((self.lockins,sub_block_size))
            self.old_signal_data = np.zeros((self.lockins,sub_block_size))
            for i1 in range(self.lockins):
                self.old_reference_data[i1,:] = new_reference_data[i1,-sub_block_size:]
                self.old_signal_data[i1,:] = new_signal_data[i1,-sub_block_size:]

        
        package['reference'] = reference_data
        package['signal'] = signal_data
        
        nblock = self.settings.get_final_output_block_size()
        time_vec = np.linspace(0,1/self.settings.get_update_frequency(),nblock)
        compensation = -1.0/self.settings.get_update_frequency()
        self.timestamps[nblock*self.block_number:nblock*(self.block_number+1)] = time.time()+time_vec+compensation
        
        time_vec = self.block_number*self.settings.get_block_size()/self.settings.get_sample_frequency()+self.base_time_vector
        package['time'] = time_vec
        self.block_number = self.block_number+1
        return package

    def do_the_ffts(self, input_package):
        """Compute the FFT's of the input signals from all tiepies. The function expects as input the output of the :attr:`TiePieLCR.get_data`. The function will return a dict with the following keys:"

        * 'reference_fft' The output of the RFFT of the reference. This is a 3 dimensional numpy array. Dimension 1 is the lockin this reference signal belongs to, dimension 2 is the sub block and dimension 3 is sample number.
        * 'signal_fft' The output of the RFFT of the reference. This is a 3 dimensional numpy array. Dimension 1 is the lockin this reference signal belongs to, dimension 2 is the sub block and dimension 3 is sample number.
        * 'reference': The voltage measured on the reference channels
        * 'signal': The voltage measured on the signal channels
        * 'time': The time since the start of the measurement for each sample in the 'reference' and 'signal'

        In some cases the algorithm will not return a dict with these keys. In that case it will return a dict with the key 'warning'. The contont of that key will explain the reason.

        :param input_package: A dict with the measured signals
        :return: A package with the obtained data
        :rtype: Dict
        """

        package = {}
        if 'reference' in input_package:
            if len(input_package['reference']) == self.lockins:
                reference = input_package['reference']
                signal = input_package['signal']
            else:
                package['warning'] = b'warning: input package with wrong size'
                return package
        else:
            package['warning'] = b'warning: empty input package'
            return package

        sub_blocks = self.settings.get_sub_blocks()
        sub_block_size = self.settings.get_sub_block_size()

        package['reference_fft'] = np.zeros((self.lockins,sub_blocks,sub_block_size+1))
        package['signal_fft'] = np.zeros((self.lockins,sub_blocks,sub_block_size+1))

        reference_cpu = reference.astype('d')
        signal_cpu = signal.astype('d')

        #print(gpu_available)

        if gpu_available:
            if len(self.window_gpu) != 2*sub_block_size:
                self.update_base_vectors()
                print('warning: gpu window frequency does not match data frequency')
                package['warning'] = b'warning: window frequency does not match data frequency'
                return package
            else:
                device = tr.device("cuda:0")
                reference_torch = tr.from_numpy(reference_cpu)
                signal_torch = tr.from_numpy(signal_cpu)
                
                reference_gpu = reference_torch.to(device)
                signal_gpu = signal_torch.to(device)

                reference_gpu_subblock = tr.zeros((self.lockins, sub_blocks, sub_block_size*2),device=device)
                signal_gpu_subblock = tr.zeros((self.lockins, sub_blocks, sub_block_size*2),device=device)
                
                for i1 in range(self.lockins):
                    for i2 in range(sub_blocks):
                        reference_gpu_subblock[i1,i2,:] = reference_gpu[i1,i2*sub_block_size:(i2+2)*sub_block_size]
                        signal_gpu_subblock[i1,i2,:] = signal_gpu[i1,i2*sub_block_size:(i2+2)*sub_block_size]

                reference_fft_gpu = tr.fft.rfft(reference_gpu_subblock*self.window_gpu)/sub_block_size
                signal_fft_gpu = tr.fft.rfft(signal_gpu_subblock*self.window_gpu)/sub_block_size

                package['reference_fft'] = reference_fft_gpu.cpu().detach().numpy()
                package['signal_fft'] = signal_fft_gpu.cpu().detach().numpy()
        else:
            reference_cpu_subblock = np.zeros((self.lockins, sub_blocks, sub_block_size*2))
            signal_cpu_subblock = np.zeros((self.lockins, sub_blocks, sub_block_size*2))
            for i1 in range(self.lockins):
                for i2 in range(sub_blocks):
                    reference_cpu_subblock[i1,i2,:] = reference_cpu[i1,i2*sub_block_size:(i2+2)*sub_block_size]
                    signal_cpu_subblock[i1,i2,:] = signal_cpu[i1,i2*sub_block_size:(i2+2)*sub_block_size]


            package['reference_fft'] = np.fft.rfft(reference_cpu_subblock*self.window_cpu)/sub_block_size
            package['signal_fft'] = np.fft.rfft(signal_cpu_subblock*self.window_cpu)/sub_block_size
        
        package['reference'] = input_package['reference']
        package['signal'] = input_package['signal']
        package['time'] = input_package['time']

        return package


    #this is where the magic happens
    def process_data(self,input_package):
        """Processes the data belonging to this instance of the TiePieLCR class. This class takes as an input the output of the :attr:`TiePieLCR.do_the_ffts` function. The function returns an dictionary with the following keys:
        
        *

        In case of an errror the dict will not return any of this keys, but instead a dict with one of these keys:

        * 'warning': meaning that issue was detected but that issue is expected to solve itself. The key contains text explaining the exact issue that was detected.
        * 'error': meaning that issue was detected but that issue is not expected to solve itself. The key contains text explaining the exact issue that was detected.        

        :param input_package: A dict with the measured signals and their fourier transforms
        :return: A dict with the obtained data
        :rtype: Dict
        """
        package = {}

        if 'warning' in input_package:
            print('found a warning in the input package')
            package['warning'] = input_package['warning']
            return package

        if 'error' in input_package:
            print('found an error in the input package')
            package['error'] = input_package['error']
            return package

        if self.inst >= input_package['reference_fft'].shape[0]:
            print('wrong number of LCRs')
            self.block_number_process= self.block_number_process+1
            package['warning'] = "input package with the wrong number of LCRs received"
            return package

        if self.settings.get_sub_blocks() != input_package['reference_fft'].shape[1]:
            print('wrong number of sublocks in package')
            self.block_number_process= self.block_number_process+1
            package['warning'] = "input package with the wrong number of LCRs received"
            return package

        if self.inst == 0 and self.state != 2:
            print('wrong state')
            self.block_number_process= self.block_number_process+1
            package['warning'] = "warning: wrong state"
            return package

        if max(self.settings.get_demodulation_tiepies()) > len(input_package['reference_fft'][:,0, 0]):
            package['error'] = b'error: demodulate frequencies table contains a tiepie# that does not exist'
            return package 

        #print("crumble13')
        if self.settings.get_final_output_block_size() == 0:
            package['error'] = b'error: final output block size not set'
            return package

        reference_fft = input_package['reference_fft']
        signal_fft = input_package['signal_fft']

        #tiepies in the received package
        tiepies_in_package = len(reference_fft[:,0, 0])

        

        package, ref_offset, sig_offset, buf1_R, buf1_I, buf2_R, buf2_I = self.decompose_fft(reference_fft,signal_fft,package)
        package = self.apply_final_low_pass_filter(buf1_R, buf1_I, buf2_R, buf2_I, package)
        package = self.integrate_demodulate_results(package)
        package = self.calculate_offsets(ref_offset, sig_offset, package)
        package = self.calculate_fft_plot_vectors(reference_fft,signal_fft,package)
        package = self.calculate_time_plot_vectors(input_package,package)
        package = self.clipping_dections(input_package, package)
        
        package['serial_numbers'] = self.settings.get_serial_numbers()

        self.block_number_process= self.block_number_process+1
        return package

    def decompose_fft(self,reference_fft,signal_fft,output_package):
        """ 
        
        *

        In case of an errror the dict will not return any of this keys, but instead a dict with one of these keys:

        * 'warning': meaning that issue was detected but that issue is expected to solve itself. The key contains text explaining the exact issue that was detected.
        * 'error': meaning that issue was detected but that issue is not expected to solve itself. The key contains text explaining the exact issue that was detected.        

        :param input_package: A dict with the measured signals and their fourier transforms
        :return: A dict with the obtained data
        :rtype: Dict
        """
        #length of the data that actually will be used
        length = int(self.settings.get_output_sub_block_size()/2)
        #length of the data will be analysed
        total_length = self.settings.get_output_block_size()
        #lenth of the data after decimation
        
        #number of frequencies in the signal
        total_freqs = self.settings.get_number_of_demodulation_freqs()
        #number of sub blocks
        sub_blocks = self.settings.get_sub_blocks()
        #reference signal that conains each demodulation frequency
        dem_tiepies = self.settings.get_demodulation_tiepies()

        
        buf1_R = np.zeros((total_length,total_freqs),dtype=np.cdouble)
        buf1_I = np.zeros((total_length,total_freqs),dtype=np.cdouble)
        buf2_R = np.zeros((total_length,total_freqs),dtype=np.cdouble)
        buf2_I = np.zeros((total_length,total_freqs),dtype=np.cdouble)
        sig_offset = np.zeros(self.settings.get_offset_block_size(),dtype=np.cdouble)
        ref_offset = np.zeros(self.settings.get_offset_block_size(),dtype=np.cdouble)

        for i2 in range(sub_blocks):

            for i1 in range(total_freqs):

                #frequency bin above the modulation frequency
                n_freq_plus = int(np.ceil(self.settings.dem_freqs[i1]/self.settings.get_df()))
                #frequency bin below the modulation frequency
                n_freq_min = int(np.floor(self.settings.dem_freqs[i1]/self.settings.get_df()-0.000001))

                #check if the frequency is high enough to do demodulation with this number of samples
                if self.settings.get_use_hf_algorithm(i1):
                    use = np.arange(n_freq_min-length+1,n_freq_plus+length)

                    #select the relevant data in the fft
                    reference_select = reference_fft[dem_tiepies[i1]-1,i2, use]
                    signal_select = signal_fft[self.inst,i2,use]
                    
                    if self.settings.bandwidth>self.settings.get_fft_sensitivity():
                        freqs = use*self.settings.get_df()
                        gain = self.settings.get_reference_gain()
                        Z = self.settings.calibration.get_measurement_z(freqs, self.settings.reference_setting,self.settings.gain_setting)
                        reference_select = -1*reference_select/gain/Z

                    sigp_x_refm_rfft= signal.convolve(signal_select,np.conj(reference_select[::-1]))
                    refp_x_refm_rfft= signal.convolve(reference_select,np.conj(reference_select[::-1]))

                    result_neg = np.arange(0,length*2-1)
                    result_pos = np.arange(length*2-1,length*4-1)#currently
                    padded_zero = np.zeros(1)

                    sigp_x_refm_fft = np.concatenate((sigp_x_refm_rfft[result_pos],padded_zero, sigp_x_refm_rfft[result_neg]))
                    refp_x_refm_fft = np.concatenate((refp_x_refm_rfft[result_pos],padded_zero, refp_x_refm_rfft[result_neg]))
                    
                    n_len = length*4
                    sig_demod = np.fft.ifft(sigp_x_refm_fft)*n_len
                    ref_demod = np.fft.ifft(refp_x_refm_fft)*n_len

                    #can't do the filtering on the ref and sig signal seperately because that fucks up with the window
                    Z_inst = sig_demod/ref_demod
                    loc = range(i2*length*2,(i2+1)*length*2)
                    buf1_R[loc,i1] = np.real(Z_inst[length:3*length])
                    buf1_I[loc,i1] = np.imag(Z_inst[length:3*length])
                else:
                    #if the frequency is too low, just do a multiplication.
                    #The final low pass filter will take care of the noise

                    #select the lowest frequencies
                    
                    oversampling_ratio = self.settings.get_output_oversample_ratio()
                    n_len = length*oversampling_ratio*2
                    
                    use = np.arange(n_len+1)
                    
                            
                    #window = np.hamming(2*n_len+1)
                    ref_windowed = reference_fft[self.inst,i2,use]#*window[n_len:]
                    sig_windowed = signal_fft[self.inst,i2,use]#*window[n_len:]
                    
                    #go back to the time domain, apply the inverse of the used window
                    if not len(self.filt_win[i1]) == len(np.fft.irfft(ref_windowed)):
                        self.update_base_vectors()
                    reference_filt = np.fft.irfft(ref_windowed)/self.filt_win[i1]*n_len
                    sig_filt = np.fft.irfft(sig_windowed)/self.filt_win[i1]*n_len

                    use = np.arange(n_len/2,3*n_len/2,1,dtype=np.int32)
                    fnyq = n_len*self.settings.get_sub_block_freq()/2
                    Wn = self.settings.dem_freqs[i1]/fnyq

                    if len(self.b_bandpass) == 0 :
                        self.update_base_vectors()
                    reference_filt2, self.zi_bandpass_ref[i1] = self.bandpass_fir_filter(reference_filt[use],self.b_bandpass[i1],self.zi_bandpass_ref[i1])
                    H_reference_filt2,self.zi_hilbert[i1] = self.low_freq_hilbert(reference_filt2,Wn,self.zi_hilbert[i1])
                    sig_filt2, self.zi_bandpass_sig[i1] = self.bandpass_fir_filter(sig_filt[use],self.b_bandpass[i1],self.zi_bandpass_sig[i1])                    

                    #store the signal for furhter filtering down the road.
                    loc = range(i2*2*length,(i2+1)*2*length)
                    use = np.arange(0,n_len,oversampling_ratio,dtype=np.int32)
                    buf1_R[loc,i1] = np.real(reference_filt2[use]*sig_filt2[use])
                    buf1_I[loc,i1] = np.real(-1*H_reference_filt2[use]*sig_filt2[use])
                    buf2_R[loc,i1] = np.real(reference_filt2[use]*reference_filt2[use])
                    buf2_I[loc,i1] = np.real(-1*H_reference_filt2[use]*reference_filt2[use])
                    

            use_bins = self.settings.get_offset_sub_block_size()
            use = range(0,use_bins+1)
            ref_offset_filt = np.fft.irfft(reference_fft[self.inst,i2,use])
            sig_offset_filt = np.fft.irfft(signal_fft[self.inst,i2,use])

            loc = range(i2*use_bins,(i2+1)*use_bins)
            use = np.arange(use_bins/2,3*use_bins/2,dtype=np.int32)
            
            ref_offset[loc] = ref_offset_filt[use]/self.offset_win[use]*use_bins
            sig_offset[loc] = sig_offset_filt[use]/self.offset_win[use]*use_bins

            gain = self.settings.calibration.reference_gain_list[self.settings.reference_setting]
            Z = self.settings.calibration.get_measurement_z(0, self.settings.reference_setting,self.settings.gain_setting)
            ref_offset[loc] = -1*ref_offset[loc]/gain/Z
        return output_package, ref_offset, sig_offset, buf1_R, buf1_I, buf2_R, buf2_I

    def apply_final_low_pass_filter(self, buf1_R, buf1_I, buf2_R, buf2_I, package):
        """Processes the data belonging to this instance of the TiePieLCR class. This class takes as an input the output of the :attr:`TiePieLCR.do_the_ffts` function. The function returns an dictionary with the following keys:
        
        *

        In case of an errror the dict will not return any of this keys, but instead a dict with one of these keys:

        * 'warning': meaning that issue was detected but that issue is expected to solve itself. The key contains text explaining the exact issue that was detected.
        * 'error': meaning that issue was detected but that issue is not expected to solve itself. The key contains text explaining the exact issue that was detected.        

        :param input_package: A dict with the measured signals and their fourier transforms
        :return: A dict with the obtained data
        :rtype: Dict
        """

        total_freqs = self.settings.get_number_of_demodulation_freqs()
        final_length= self.settings.get_final_output_block_size()


        n = len(buf1_R[:,0])
        fs = self.settings.get_output_sample_freq()
        Wn = self.settings.bandwidth/fs*2
        downsampling_rate = TiePieLCR_settings.calculate_downsapling_rate(n,Wn)
        new_demodulate_result = np.zeros((final_length,total_freqs),dtype=np.cdouble)
        for i1 in range(total_freqs):
            if i1 >= len(self.zi_buf1_R):
                package['warning'] = b'warning: needed a reset buffers'
                return package
            #compute initial condition if none exists.

            

            if self.settings.get_use_hf_algorithm(i1):
                temp_result_real_sig,self.zi_buf1_R[i1] = signal.sosfilt(self.sos_decimate[i1],buf1_R[:,i1], zi=self.zi_buf1_R[i1])
                temp_result_imag_sig,self.zi_buf1_I[i1] = signal.sosfilt(self.sos_decimate[i1],buf1_I[:,i1], zi=self.zi_buf1_I[i1])

                temp_result_real_sig = temp_result_real_sig[0::downsampling_rate]
                temp_result_imag_sig = temp_result_imag_sig[0::downsampling_rate]

                division = temp_result_real_sig + 1j*temp_result_imag_sig
            else:
                

                temp_result_real_sig,self.zi_buf1_R[i1] = signal.sosfilt(self.sos_decimate[i1],buf1_R[:,i1], zi=self.zi_buf1_R[i1])
                temp_result_imag_sig,self.zi_buf1_I[i1] = signal.sosfilt(self.sos_decimate[i1],buf1_I[:,i1], zi=self.zi_buf1_I[i1])
                temp_result_real_ref,self.zi_buf2_R[i1] = signal.sosfilt(self.sos_decimate[i1],buf2_R[:,i1], zi=self.zi_buf2_R[i1])
                temp_result_imag_ref,self.zi_buf2_I[i1] = signal.sosfilt(self.sos_decimate[i1],buf2_I[:,i1], zi=self.zi_buf2_I[i1])

                temp_result_real_sig = temp_result_real_sig[0::downsampling_rate]
                temp_result_imag_sig = temp_result_imag_sig[0::downsampling_rate]
                temp_result_real_ref = temp_result_real_ref[0::downsampling_rate]
                temp_result_imag_ref = temp_result_imag_ref[0::downsampling_rate]

                ref_x_sig = temp_result_real_sig+1j*temp_result_imag_sig
                ref_x_ref = temp_result_real_ref+1j*temp_result_imag_ref
                division = ref_x_sig/ref_x_ref

            if self.settings.bandwidth < self.settings.get_fft_sensitivity():
                gain = self.settings.calibration.reference_gain_list[self.settings.reference_setting]
                Z = self.settings.calibration.get_measurement_z(self.settings.dem_freqs[i1], self.settings.reference_setting,self.settings.gain_setting)
                temp = -division*gain
                new_demodulate_result[:,i1] = Z*temp
            else:
                new_demodulate_result[:,i1] = division

            
        
        if new_demodulate_result.shape[1] != self.demodulate_save_buffer.shape[1]:
                package['warning'] = b'warning: needed to reset buffers'
                return package
        #store the result in order to be able to save it in the end.
        self.demodulate_save_buffer[final_length*self.block_number_process:final_length*(self.block_number_process+1),:] = new_demodulate_result[:,:]
        return package

    def integrate_demodulate_results(self,package):
        #lenth of the data after decimation
        final_length= self.settings.get_final_output_block_size()
        #number of frequencies in the signal
        total_freqs = self.settings.get_number_of_demodulation_freqs()

        if self.demodulate_save_buffer.shape[1] != total_freqs:
            self.reset_buffers()


        tblock = self.settings.get_block_size()/self.settings.get_sample_frequency()
        dt = tblock/final_length
        plot_time = self.settings.get_plot_time()
        plot_blocks = self.settings.get_plot_blocks()
        plot_points = self.settings.get_demodulate_plot_points()
        if self.block_number_process+1 > self.settings.get_plot_blocks():
            use = np.linspace(final_length*(self.block_number_process+1-plot_blocks),final_length*(self.block_number_process+1)-1,plot_points,dtype=int)
            package['plot_demodulate_time'] = use*dt
        else:
            fake_points = int(plot_points*(plot_blocks-self.block_number_process)/plot_blocks)
            fake_time = np.linspace(-1,0,fake_points)*plot_time*(plot_blocks-self.block_number_process)/plot_blocks
            fake_data = np.zeros(fake_points, dtype=np.int32)
            real_data = np.linspace(0,final_length*(self.block_number_process+1)-1,plot_points-fake_points,dtype=int)
            use = np.concatenate((fake_data,real_data))
            plot_time = np.concatenate((fake_time,real_data*dt))
            package['plot_demodulate_time'] = plot_time

        #make sure there is not to much data to plot
        package['plot_demodulate_impedance'] = self.demodulate_save_buffer[use,:]
        package['plot_demodulate_1'] = np.zeros((len(use),total_freqs))
        package['plot_demodulate_2'] = np.zeros((len(use),total_freqs))

        #calculate the impedance values by integrating
        nend = final_length*(self.block_number_process+1)-1
        nint = int(self.settings.get_integration_time()/dt)
        if nint>final_length*(self.block_number_process+1):
            use_int = np.arange(0,nend)
        else:
            use_int = np.arange(nend-nint,nend)
        
        
        if nend == 0:
            package['int_timestamp'] = 0
        else:
            package['int_timestamp'] = self.timestamps[use_int[int(len(use_int)/2)]]
        package['int_demodulate_1'] = np.zeros(total_freqs)
        package['int_demodulate_2'] = np.zeros(total_freqs)
        package['int_demodulate_1_std'] = np.zeros(total_freqs)
        package['int_demodulate_2_std'] = np.zeros(total_freqs)
        package['int_demodulate_1_noise_density'] = np.zeros(total_freqs)
        package['int_demodulate_2_noise_density'] = np.zeros(total_freqs)
        package['int_demodulate_1_error'] = np.zeros(total_freqs)
        package['int_demodulate_2_error'] = np.zeros(total_freqs)
        for i1 in range(total_freqs):
            temp_f = self.settings.dem_freqs[i1]
            if self.settings.get_impedance_format() == 'XY':
                package['plot_demodulate_1'][:,i1] = np.real(self.demodulate_save_buffer[use,i1])
                package['plot_demodulate_2'][:,i1] = np.imag(self.demodulate_save_buffer[use,i1])
                package['int_demodulate_1'][i1] = np.mean(np.real(self.demodulate_save_buffer[use_int,i1]))
                package['int_demodulate_2'][i1] = np.mean(np.imag(self.demodulate_save_buffer[use_int,i1]))
                package['int_demodulate_1_std'][i1] = np.std(np.real(self.demodulate_save_buffer[use_int,i1]))
                package['int_demodulate_2_std'][i1] = np.std(np.imag(self.demodulate_save_buffer[use_int,i1]))
            elif self.settings.get_impedance_format() == 'RpCp':
                a = np.real(self.demodulate_save_buffer[use,i1])
                b = np.imag(self.demodulate_save_buffer[use,i1])
                package['plot_demodulate_1'][:,i1] = (b**2+a**2)/a
                package['plot_demodulate_2'][:,i1] = -b/package['plot_demodulate_1'][:,i1]/a/2/np.pi/temp_f
                a = np.real(self.demodulate_save_buffer[use_int,i1])
                b = np.imag(self.demodulate_save_buffer[use_int,i1])
                package['int_demodulate_1'][i1] = np.mean((b**2+a**2)/a)
                package['int_demodulate_2'][i1] = np.mean(-b/package['int_demodulate_1'][i1]/a/2/np.pi/temp_f)
                package['int_demodulate_1_std'][i1] = np.std((b**2+a**2)/a)
                package['int_demodulate_2_std'][i1] = np.std(-b/package['int_demodulate_1'][i1]/a/2/np.pi/temp_f)
            elif self.settings.get_impedance_format() == 'RsCs':
                package['plot_demodulate_1'][:,i1] = np.real(self.demodulate_save_buffer[use,i1])
                package['plot_demodulate_2'][:,i1] = -1/np.imag(self.demodulate_save_buffer[use,i1])/2/np.pi/temp_f
                package['int_demodulate_1'][i1] = np.mean(np.real(self.demodulate_save_buffer[use_int,i1]))
                package['int_demodulate_2'][i1] = np.mean(-1/np.imag(self.demodulate_save_buffer[use_int,i1])/2/np.pi/temp_f)
                package['int_demodulate_1_std'][i1] = np.std(np.real(self.demodulate_save_buffer[use_int,i1]))
                package['int_demodulate_2_std'][i1] = np.std(-1/np.imag(self.demodulate_save_buffer[use_int,i1])/2/np.pi/temp_f)
            elif self.settings.get_impedance_format() == 'ZPhi':
                package['plot_demodulate_1'][:,i1] = np.abs(self.demodulate_save_buffer[use,i1])
                package['plot_demodulate_2'][:,i1] = np.angle(self.demodulate_save_buffer[use,i1])
                package['int_demodulate_1'][i1] = np.mean(np.abs(self.demodulate_save_buffer[use_int,i1]))
                package['int_demodulate_2'][i1] = np.mean(np.angle(self.demodulate_save_buffer[use_int,i1]))
                package['int_demodulate_1_std'][i1] = np.std(np.abs(self.demodulate_save_buffer[use_int,i1]))
                package['int_demodulate_2_std'][i1] = np.std(np.angle(self.demodulate_save_buffer[use_int,i1]))
            else:
                print('error: unknown impedance format')
            package['int_demodulate_1_noise_density'][i1] = package['int_demodulate_1_std'][i1]/np.sqrt(self.settings.get_demodulation_bandwidth())
            package['int_demodulate_2_noise_density'][i1] = package['int_demodulate_2_std'][i1]/np.sqrt(self.settings.get_demodulation_bandwidth())
            package['int_demodulate_1_error'][i1] = package['int_demodulate_1_noise_density'][i1]/np.sqrt(self.settings.get_integration_time())
            package['int_demodulate_2_error'][i1] = package['int_demodulate_2_noise_density'][i1]/np.sqrt(self.settings.get_integration_time())
        
        return package
        

    def calculate_offsets(self, ref_offset, sig_offset, package):
        final_length= self.settings.get_final_output_block_size()
        tblock = self.settings.get_block_size()/self.settings.get_sample_frequency()
        dt = tblock/final_length

        fs = self.settings.get_offset_sample_frequency()
        Wn = self.settings.offset_bandwidth/fs*2
        final_ref_offset,self.zi_ref_offset_downsample = signal.sosfilt(self.sos_offset,ref_offset, zi=self.zi_ref_offset_downsample)
        final_sig_offset,self.zi_sig_offset_downsample = signal.sosfilt(self.sos_offset,sig_offset, zi=self.zi_sig_offset_downsample)
        n = len(ref_offset)
        downsampling_rate = TiePieLCR_settings.calculate_downsapling_rate(n,Wn)
        final_ref_offset = final_ref_offset[0::downsampling_rate]
        final_sig_offset = final_sig_offset[0::downsampling_rate]

        offset_length = self.settings.get_final_offset_block_size()
        self.ref_offset_save_buffer[offset_length*self.block_number_process:offset_length*(self.block_number_process+1)] = final_ref_offset  
        self.sig_offset_save_buffer[offset_length*self.block_number_process:offset_length*(self.block_number_process+1)] = final_sig_offset

        #calculate the offset values
        nend = offset_length*(self.block_number_process+1)
        nint = int(self.settings.offset_integration_time/dt)
        if nint>offset_length*(self.block_number_process+1):
            use_int = np.arange(0,nend)
        else:
            use_int = np.arange(nend-nint,nend)
        package['int_ref_offset'] = np.mean(np.real(self.ref_offset_save_buffer[use_int]))
        package['int_sig_offset'] = np.mean(np.real(self.sig_offset_save_buffer[use_int]))
        return package

    def calculate_fft_plot_vectors(self,reference_fft,signal_fft,output_package):
        mask = (self.base_freq_vector>self.settings.fmin_plot) & (self.base_freq_vector<self.settings.fmax_plot)
        plot_fft_points = np.where(mask)
        decimator = int(np.ceil(len(plot_fft_points)/self.settings.fft_plot_points))
        plot_fft_points = plot_fft_points[::decimator]

        df = self.settings.get_sample_frequency()/self.settings.get_block_size()
        output_package['plot_reference_fft'] = np.squeeze(np.abs(reference_fft[self.inst,-1,plot_fft_points])/2/df**0.5)
        output_package['plot_signal_fft'] = np.squeeze(np.abs(signal_fft[self.inst,-1,plot_fft_points])/2/df**0.5)
        output_package['plot_frequency'] = self.base_freq_vector[plot_fft_points]

        return output_package

    def calculate_time_plot_vectors(self,input_package,output_package):
        if self.settings.get_f_fun() > 0:
            plot_time = self.settings.get_plot_periods()/self.settings.get_f_fun()
        else:
            plot_time = 99999999999
        t0 = self.settings.get_block_size()/self.settings.get_sample_frequency()
        if plot_time > t0 :
            plot_time = t0
            #print("trying to plot more data as that is available in one block")

        start_point = 0
        stop_point = np.round((plot_time)*self.settings.get_sample_frequency())-1
        plot_points = np.linspace(start_point,stop_point,self.settings.get_time_plot_points(),dtype=int)
        
        output_package['plot_reference'] = input_package['reference'][self.inst][plot_points]
        output_package['plot_signal'] = input_package['signal'][self.inst][plot_points]
        output_package['plot_time'] = input_package['time'][plot_points]
        return output_package
        

    def clipping_dections(self,input_package, output_package):
        ref_max = np.max(input_package['reference'][self.inst,:])
        ref_min = np.min(input_package['reference'][self.inst,:])
        sig_max = np.max(input_package['signal'][self.inst,:])
        sig_min = np.min(input_package['signal'][self.inst,:])

        ref_gain = np.abs(float(self.settings.get_reference_gain()))
        sig_gain = self.settings.get_gain_value()
        ref_range = self.settings.get_reference_scope_range_value()
        sig_range = self.settings.get_signal_scope_range_value()

        margin = 0.99

        integration_blocks = self.settings.get_integration_time()*self.settings.get_update_frequency()
        if ref_min < -ref_gain*ref_range*margin or ref_max > ref_gain*ref_range*margin or output_package['plot_demodulate_time'][-1] < self.settings.get_integration_time():
            output_package['ref_clipping_now'] = True
            output_package['ref_clipping_during_integration'] = True
            self.clipping_counter_ref = integration_blocks
        else: 
            output_package['ref_clipping_now'] = False
            if self.clipping_counter_ref <= 0:
                output_package['ref_clipping_during_integration'] = False
            else:
                output_package['ref_clipping_during_integration'] = True
                self.clipping_counter_ref = self.clipping_counter_ref - 1

        if sig_min < -sig_gain*sig_range*margin or sig_max > sig_gain*sig_range*margin or output_package['plot_demodulate_time'][-1] < self.settings.get_integration_time():
            output_package['sig_clipping_now'] = True
            output_package['sig_clipping_during_integration'] = True
            self.clipping_counter_sig = integration_blocks
        else:
            output_package['sig_clipping_now'] = False
            if self.clipping_counter_sig <= 0:
                output_package['sig_clipping_during_integration'] = False
            else:
                output_package['sig_clipping_during_integration'] = True
                self.clipping_counter_sig = self.clipping_counter_sig - 1

        return output_package

    @staticmethod
    def design_bandpass_fir_filter(n,Wn,width):
        taps = int(1/width*5)+1    
        filter_b = signal.firwin(taps,(Wn-width,Wn+width),pass_zero='bandpass')
        filter_b = signal.minimum_phase(filter_b)
        return filter_b

    @staticmethod
    def bandpass_fir_filter(x,filter_b,prev_values):
        n = len(x)
        total_n = len(filter_b)+n-1      
        if len(prev_values) != total_n:
            prev_values =  np.ones(total_n)*x[0]
        prev_values[:-n] = prev_values[n:]
        prev_values[-n:] = x
        y = signal.convolve(prev_values,filter_b,mode ='valid',method='fft')
        return y,prev_values

    @staticmethod
    def low_freq_hilbert(x,Wn,prev_values):
        n = len(x)

        total_n = int(1/Wn*10)
        if total_n < n:
            total_n = n

        if len(prev_values) == 0:
            prev_values =  np.ones(total_n)*x[0]
        prev_values[:-n] = prev_values[n:]
        prev_values[-n:] = x
        y = np.imag(signal.hilbert(prev_values))
        return y[-n:],prev_values

    def start_measurement(self,all_settings):
        package ={}
        package = self.open_scope()
        if 'error' in package:
            return package
        if not self.open_gen():
            package['error'] = b"Failed to open generator. This error should not occur."
            return package

        for i1 in range(self.lockins):
            if all_settings[i1].no_frontend == False:
                if not self.select_reference(i1,all_settings[i1].reference_setting,all_settings[i1].gain_setting):
                    serial_numbers = self.settings.get_serial_numbers()
                    package['error'] = b"Failed to set reference of the frontend connected to TiePie " + str(serial_numbers[i1]).encode() + b". Is the frontend turned on and connected?"
                    return package
                if not self.select_LCR_gain(i1,all_settings[i1].reference_setting,all_settings[i1].gain_setting):
                    package['error'] = b"Failed to select gain!"
                    return package
            self.start_awg(i1,all_settings[i1])

        self.start_stream(all_settings)
        return package
    
    def stop_measurement(self):
        self.stop_stream()
        for i1 in range(self.lockins):
            self.stop_gen(i1,True)

    def get_stored_data(self):
        package = {}
        
        if self.block_number_process == 0:
            package['empty'] = True
            return package
        
        gc.collect()
        n = self.block_number_process*self.settings.get_final_output_block_size()
        package['demodulation_data'] = self.demodulate_save_buffer[:n,:]
        package['demodulation_time'] = self.settings.get_demodulation_time_vector(self.block_number_process)
        package['timestamps'] = self.timestamps[:n]

        n_offset = self.block_number_process*self.settings.get_final_offset_block_size()
        package['ref_offset'] = self.ref_offset_save_buffer[:n_offset]
        package['sig_offset'] = self.sig_offset_save_buffer[:n_offset]
        package['offset_time'] = self.settings.get_offset_time_vector(self.block_number_process)

        return package

    def stop_gen(self,inst,disable_output):
        if len(self.gen) == self.lockins:
            # Stop generator:
            self.gen[inst].stop()
            if disable_output:
                # Disable output:
                self.gen[inst].output_on = False

    def stop_stream(self):
        if self.state == 2 or self.state == 4:
            if self.scp:
                self.state = 3
                self.scp.stop()
                self.state = 0    


    def close(self):
        return 0