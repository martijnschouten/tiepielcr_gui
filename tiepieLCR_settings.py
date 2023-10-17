"""
.. module:: TiePieLCR_settings
   :synopsis: This class contains all settings of a single TiePieLCR
.. moduleauthor:: Martijn Schouten <github.com/martijnschouten>
"""

from re import S
import yaml
import io
import numpy as np
import math
from array import array
from TiePieLCR_calibration import TiePieLCR_calibration
try:
    import torch as tr
    from torch import nn, optim
    test = tr.zeros(3)
    gpu_available = True
except Exception as e:
    gpu_available = False

class TiePieLCR_settings:
    

    """When the index of the lowest fft bin that contains the signal is below this value,  the low freqeuncy demodulation algorithm will be used """

    def __init__(self):
        self.max_sample_frequency = 6.25e6
        self.sample_frequency = 6.25e6
    
        self.Vmax = 2.2

        self.enabled = [True,True]

        self.gen_offset = 0

        self.gen_samples_max = 1e6
        self.gen_sample_freq_max = 240e6
        self.gen_sample_freq = 0.7e6
        self.gen_samples_min = 0.5e6
        self.gen_amplitude = 0.7
        self.gen_freqs = [5000]
        self.gen_weights = [1.0]
        self.gen_phases = [0]

        self.dem_freqs = [5000]
        self.dem_tiepies = [1]

        self.dem_is_gen = True

        self.plot_points = 200
        self.plot_periods = 3
        self.fft_plot_points = 200
        self.demodulate_plot_points = 200
        self.fmin_plot = 100
        self.fmax_plot = 1000000

        self.side_lob_n = 7

        self.inst = 1

        #these values are just for displaying, don't have to be calibrated.
        self.reference_setting = 1
        self.gain_setting = 0
        

        self.scope_ranges = [2,2]
        self.scope_range_list = [0.2, 0.4 ,0.8, 2, 4,8]
        self.scope_range_name_list = ['0.2','0.4','0.8','2','4','8']


        self.scope_coupling_name_list = ['AC', 'DC']
        self.scope_couplings_list = [2,1]
        self.scope_couplings = [1,1]
        self.scope_auto_ranging = [True,True]

        self.fs_name_list = ['6250000','3125000','1562500','781250']
        self.fs_list = [6250000,3125000,1562500,781250]
        self.fs_setting = 1

        self.update_freq_name_list = ['1 Hz','2 Hz','5 Hz','10 Hz','25 Hz']
        self.update_freq_list = [1,2,5,10,25]
        self.update_freq_setting = 3

        self.sub_blocks_name_list = ['1','2','5','10','20','50']
        self.sub_blocks_list = [1,2,5,10,20,50]
        self.sub_blocks_setting = 0

        self.restart_required = False
        self.reference_update_required = False
        self.gain_update_required = False
        self.multisine_update_required = False
        self.gen_restart_required = False
        self.base_vector_update_required = False
        self.gen_offset_update_required = False
        self.gen_amplitude_update_required = False
        self.dem_freq_update_required = False

        self.f_fun = 0
        self.f_min = 0
        self.f_max = 0

        self.bandwidth = 20
        self.plot_time = 5
        self.impedance_format = 0
        self.impedance_formats = ['XY','RpCp','RsCs','ZPhi']
        self.impedance_format_unit1 = ['Ω','Ω','Ω','Ω']
        self.impedance_format_unit2 = ['Ω','F','F','rad']
        self.impedance_format_label1 = ['X','Rp','Rs','Z']
        self.impedance_format_label2 = ['Y','Cp','Cs','Phi']

        self.output_block_size = 0
        self.save_memory = 5e6
        self.offset_save_memory = 1e7

        self.integration_time = 1
        self.fft_sensitivity = 0
        self.settings_dict = {}

        self.offset_bandwidth = 20
        self.offset_integration_time = 1

        self.output_oversample_ratio = 3
        self.real_time_mode = False
        self.version = 'V1.0.3'

        self.serial_numbers = []

        self.neglect_bins = 20

        self.no_frontend = False

        self.frontend_number = 1

        self.calibration = None

    def get_reference_unit(self):
        """Get the unit of the reference signal.

        :return: A string with the abbreviation of the unit. i.e. 'V', 'A' or 'C'
        :rtype: String
        """
        return self.calibration.reference_unit_list[self.reference_setting]

    def get_reference_offset_unit(self):
        """Get the unit of the reference offset.

        :return: A string with the abbreviation of the unit. i.e. 'V' or 'A'
        :rtype: String
        """
        return self.calibration.reference_offset_unit_list[self.reference_setting]
    
    def get_reference_scope_range_index(self):
        """Get which element of the scope range list is selected for the reference channel

        :return: The current index
        :rtype: Int
        """
        return self.scope_ranges[0]

    def set_reference_scope_range(self,range):
        """Set which element of the scope range list is selected for the reference channel. The scope range list can be obtained using :func:`TiePieLCR_settings.TiePieLCR_settings.get_scope_range_list`.

        :return: Nothing
        :rtype: None
        """
        self.scope_ranges[0] = range
        self.restart_required = True
    
    def get_signal_scope_range_index(self):
        """Get which element of the scope range list is selected for the signal channel

        :return: The current index
        :rtype: Int
        """
        return self.scope_ranges[1]

    def set_signal_scope_range(self,range):
        """Set which element of the scope range list is selected for the signal channel. The scope range list can be obtained using :func:`TiePieLCR_settings.TiePieLCR_settings.get_scope_range_list`.

        :return: Nothing
        :rtype: None
        """
        self.scope_ranges[1] = range
        self.restart_required = True

    def get_scope_range_list(self):
        """Get a list of the different gains of the scope that can be used.

        :return: The current index
        :rtype: Int
        """
        return self.scope_range_list

    def get_reference_scope_range_value(self):
        """Get the voltage range of the channel of the scope that is used to measure the reference

        :return: The range as a float. i.e. 0.2 or 2
        :rtype: Float
        """
        return self.scope_range_list[self.scope_ranges[0]]
    
    def get_signal_scope_range_value(self):
        """Get the voltage range of the channel of the scope that is used to measure the signal

        :return: The range as a float. i.e. 0.2 or 2
        :rtype: Float
        """
        return self.scope_range_list[self.scope_ranges[1]]

    def get_reference_scope_coupling(self):
        """Get the index of the coupling of the channel of the scope that is used to measure the reference

        :return: Index in scope_coupling_name_list
        :rtype: Int
        """
        return self.scope_couplings[0]

    def set_reference_scope_coupling(self,coupling):
        """Set the index of the coupling of the channel of the scope that is used to measure the reference

        :param coupling: Index in the list returned by :func:`TiePieLCR_settings.TiePieLCR_settings.get_scope_coupling_name_list`
        :type coupling: Int
        :return: Nothing
        :rtype: none
        """
        self.scope_couplings[0] = coupling
        self.restart_required = True
    
    def get_scope_coupling_name_list(self):
        """Get a list with the different coupling options

        :return: A list containing the coupling options
        :rtype: Int
        """
        return self.scope_range_name_list

    def get_signal_scope_coupling(self):
        """Get the index of the coupling of the channel of the scope that is used to measure the signal

        :return: 0 for AC, 1 for DC
        :rtype: Int
        """
        return self.scope_couplings[1]
    
    def set_signal_scope_coupling(self,coupling):
        """Set the index of the coupling of the channel of the scope that is used to measure the signal

        :param coupling: Index in the list returned by :func:`TiePieLCR_settings.TiePieLCR_settings.get_scope_coupling_name_list`
        :type coupling: Int
        :return: Nothing
        :rtype: none
        """
        self.scope_couplings[1] = coupling
        self.restart_required = True

    def get_gain_value(self):
        """Get the current gain of the instrumentation amplifier

        :return: The gain
        :rtype: float
        """
        return self.calibration.gain_list[self.gain_setting]

    def get_gain_setting(self):
        """Get the index of the currently selected gain setting. The list of possible gain can be obtained using :func:`TiePieLCR_settings.TiePieLCR_settings.get_gain_name_list`

        :return: The index
        :rtype: Int
        """
        return self.gain_setting

    def set_LCR_gain(self,new_gain):
        """Set the instrumentation amplifier gain to one of the options in the list of possible gains. The list of possible gain can be obtained using :func:`TiePieLCR_settings.TiePieLCR_settings.get_gain_name_list`

        :param new_gain: The index in the list
        :type new_gain: Int
        :return: Nothing
        :rtype: None
        """
        print("running set LCR gain in %s"%(self.inst))
        self.gain_setting = new_gain
        self.gain_update_required = True
    
    def get_gain_name_list(self):
        """Get a list of possible gains for the instrumentation amplifier.

        :return: List of gains
        :rtype: List of floats
        """
        return self.gain_name_list

    def get_reference_gain(self):
        """Get the gain of the transimpedance amplifier

        :return: The gain of the transimpedance amplifier in A/V or C/V
        :rtype: float
        """
        return self.calibration.reference_gain_list[self.reference_setting]
    
    def get_reference_setting(self):
        """Get the index of the currently selected reference setting. The list of possible references can be obtained using :func:`TiePieLCR_settings.TiePieLCR_settings.get_gain_name_list`

        :return: The index
        :rtype: Int
        """
        return self.reference_setting

    def set_reference(self,new_reference):
        """Set the transimpedance amplifier to one of the options in the list of possible options. The list of possible transimpedance amplifiers can be obtained using :func:`TiePieLCR_settings.TiePieLCR_settings.get_reference_name_list`

        :param new_gain: The index in the list
        :type new_gain: Int
        :return: Nothing
        :rtype: None
        """
        self.reference_setting = new_reference
        self.reference_update_required = True

    def get_reference_name_list(self):
        """Get a list of available transimpedance amplifiers 

        :return: List of transimpedance amplifiers
        :rtype: List of floats
        """
        return self.reference_name_list
    
    def get_gen_amplitude(self):
        """Get the maximum amplitude of the multi-sine used for the excitation. The excitation signal will determine the voltage on HcurV and when multiplied with 370uA/V also the current through HcurI. The inverse of the excitation signal will determine the voltage on nHcurV and when multiplied with 370uA/V also the current through nHcurI. This is equivalent to the excitation amplitude in the interface.

        :return: The amplitude
        :rtype: float
        """
        return self.gen_amplitude

    def get_gen_offset(self):
        """Get the offset of the excitation. The excitation signal will determine the voltage on HcurV and when multiplied with 370uA/V also the current through HcurI. The inverse of the excitation signal will determine the voltage on nHcurV and when multiplied with 370uA/V also the current through nHcurI. This is equivalent to the excitation amplitude in the interface.

        :return: The offset
        :rtype: float
        """
        return self.gen_offset

    def get_multisine_freqs(self):
        """Get a list with the frequencies in the multisine that are used as an excitation signal.

        :return: The frequencies
        :rtype: list of floats
        """
        return self.gen_freqs
    
    def get_multisine_weights(self):
        """Get a list with weights of each of the frequencies in the multisine. The weights are defined relative to the output of :func:`TiePieLCR_api.TiePieLCR_api.get_gen_amplitude` .

        :return: The weights
        :rtype: list of floats
        """
        return self.gen_weights

    def get_multisine_phases(self):
        """Get a list with phases of each of the frequencies in the multisine. The weights are defined relative to the output of :func:`TiePieLCR_api.TiePieLCR_api.get_gen_amplitude` .

        :return: The phases
        :rtype: list of floats
        """
        return self.gen_phases

    def get_demodulation_freqs(self):
        """Get a list with the frequencies at which the measured singal is demodulated.

        :return: The frequencies
        :rtype: list of floats
        """
        return self.dem_freqs

    def get_demodulation_tiepies(self):
        """Get a list with tiepie the reference signal of a specific frequency can be found.

        :return: The tiepie
        :rtype: list of ints
        """
        return self.dem_tiepies        

    def get_plot_periods(self):
        """Get the number of repetitions that are shown in the time plot in the left top of the interface.

        :return: The number of periods
        :rtype: float
        """
        return self.plot_periods

    def set_plot_periods(self,periods):
        """Set the number of repetitions that are shown in the time plot in the left top of the interface.

        :param periods: The number of periods
        :type periods: float
        :return: Nothing
        :rtype: None
        """
        self.plot_periods = periods

    def get_minimum_plot_frequency(self):
        """Get the minimum frequency that is shown in the frequency plots on the top right.

        :return: The frequency
        :rtype: float
        """
        return self.fmin_plot
    
    def get_maximum_plot_frequency(self):
        """Get the maximum frequency that is shown in the frequency plots on the top right.

        :return: The frequency
        :rtype: float
        """
        return self.fmax_plot
    
    def get_impedance_format(self):
        """Get the complex impedance measured by the LCR can be represented in different ways:

        * XY: As a complex and an imaginary part
        * RpCp: As a capacitor and a resistor in parallel
        * RsCs: As a capacitor and a reisistor in series
        * ZPhi: As an absolute value and a phase 

        :return: The used format as a string i.e. 'XY' or 'RpCp'
        :rtype: String
        """
        return self.impedance_formats[self.impedance_format]
        
    
    def set_impedance_format(self,value):
        """Set the complex impedance measured by the LCR can be represented in different ways:

        * XY: As a complex and an imaginary part
        * RpCp: As a capacitor and a resistor in parallel
        * RsCs: As a capacitor and a reisistor in series
        * ZPhi: As an absolute value and a phase 

        :param value: The used format as a string i.e. 'XY' or 'RpCp'
        :type value: String
        :return: Nothing
        :rtype: None
        """
        self.impedance_format = value

    def get_impedance_format_unit1(self):
        """Get the unit of the first value of the impedance format

        :return: The used format as a string i.e. 'F'
        :rtype: String
        """
        return self.impedance_format_unit1[self.impedance_format]

    def get_impedance_format_unit2(self):
        """Get the unit of the second value of the impedance format

        :return: The used format as a string i.e.'F'
        :rtype: String
        """
        return self.impedance_format_unit2[self.impedance_format]

    def get_impedance_format_label1(self):
        """Get the label of the first value of the impedance format

        :return: The used format as a string i.e. 'F'
        :rtype: String
        """
        return self.impedance_format_label1[self.impedance_format]

    def get_impedance_format_label2(self):
        """Get the label of the second value of the impedance format

        :return: The used format as a string i.e. 'F'
        :rtype: String
        """
        return self.impedance_format_label2[self.impedance_format]
    
    def get_serial_number(self,instance):
        return self.serial_numbers[instance]

    def get_serial_numbers(self):
        return self.serial_numbers

    def set_serial_numbers(self,serial_numbers):
        self.serial_numbers = serial_numbers


    def get_number_of_demodulate_freqs(self):
        """Get the number of frequencies inside the multisine

        :return: The used format as a string i.e. 'F'
        :rtype: Int
        """
        return len(self.dem_freqs)

    def get_demodulation_bandwidth(self):
        return self.bandwidth
        """Get the bandwidth of the demodulated signals. This is the bandwidth of the signals in the bottom two graphs of the interface

        :return: The bandwidth
        :rtype: Float
        """

    def get_time_plot_points(self):
        """Get the number of points that are plotted in the time plots. Setting this to a to high value will make things slow, settings this to a too high value will cause aliasing.

        :return: The number of points
        :rtype: Int
        """
        return self.plot_points

    def get_offset_integration_time(self):
        """Get the time over which the offset signals will be avaraged to compute the offset displayed in the interface and the offset obtain using :func:`TiePieLCR_api.TiePieLCR_api.get_impedance`.

        :return: The amount of time
        :rtype: Int
        """
        return self.offset_integration_time

    def set_offset_integration_time(self,new_offset_integration):
        """Over how much time the the offset signals are averaged to come to the displayed impedances.

        :param bandwidth: The amount of time
        :type bandwidth: Float
        :return: Nothing
        :rtype: None
        """
        self.offset_integration_time = new_offset_integration
        self.restart_required = True
        self.base_vector_update_required = True
    
    def get_offset_bandwidth(self):
        """Get the bandwidth of the offset signals. The signal computed using this bandwidth can be found in the stored mat file`.

        :return: The bandwidth
        :rtype: Float
        """
        return self.offset_bandwidth
    
    def get_use_hf_algorithm(self,i1):
        n_freq_min = int(np.floor(self.dem_freqs[i1]/self.get_df()-0.000001))
        
        length = int(self.get_output_sub_block_size()/2)

        #check if the frequency is high enough to do demodulation with this number of samples
        if n_freq_min > length + 1 +self.neglect_bins:
            return True
        else:
            return False

    def set_offset_bandwidth(self,new_offset_bandwidth):
        """Set the bandwidth with which the offset is calculated and what will be the bandwidth of the offset signals in the mat file.

        :param bandwidth: The bandwidth
        :type bandwidth: Float
        :return: Nothing
        :rtype: None
        """
        self.offset_bandwidth = new_offset_bandwidth
        self.restart_required = True
        self.base_vector_update_required = True

    def optimise_crest(self):
        
        if not gpu_available:
            return False

        freqs = self.get_multisine_freqs()
        weights = self.get_multisine_weights()
        fs = self.get_gen_sample_freq()
        n = self.get_gen_samples()
        dt = 1.0/fs
        print("Started looking for optimimum crest factor")
        
        device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")
        print(tr.cuda.is_available())
        print(device)
        phases_tr = tr.rand(len(freqs),dtype=tr.float32,requires_grad=True, device=device)

        t = np.arange(0,n*dt-dt/2,dt)
        t_tr =  tr.from_numpy(t)
        t_tr = t_tr.type(tr.float32)
        t_tr = t_tr.to(device)

        freqs_tr =  tr.from_numpy(freqs)
        freqs_tr = freqs_tr.type(tr.float32)
        freqs_tr = freqs_tr.to(device)

        weights_tr =  tr.from_numpy(weights)
        weights_tr = weights_tr.type(tr.float32)
        weights_tr = weights_tr.to(device)

        criterion = nn.BCELoss()
        learning_rate = 0.01
        optimizer = optim.Adam([phases_tr], learning_rate)

        target = tr.zeros(1).to(device)
        target = tr.squeeze(target)

        self.max_epochs = 1000
        for epoch in range(self.max_epochs):
            Y_pred = self.crest_factor_cost_function(phases_tr, freqs_tr,weights_tr,t_tr,device)
            
            Y_pred = tr.squeeze(Y_pred)
            
            train_loss = criterion(Y_pred, target)

            if epoch%100==0:
                result = Y_pred.cpu().detach().numpy()
                self.progress = epoch/self.max_epochs
                print(result)
                print(self.progress)
            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()

        self.gen_phases  = phases_tr.cpu().detach().numpy()
        print("Found optimimum crest factor:")
        print(self.get_multisine_crest_factor())
        print('Using phases:')
        print(self.gen_phases)

    def crest_factor_cost_function(self,phases,freqs,weights,t,device):
        if not gpu_available:
            return False
        y = tr.zeros(len(t),device=device)
        pi = tr.ones(1)*3.14
        pi = pi.to(device=device)
        for i1 in range(len(freqs)):
            y = y + weights[i1]*tr.sin(2*pi*freqs[i1]*t+phases[i1])
        ampl = tr.max(tr.abs(y))/tr.sum(weights)
        return ampl

    def get_plot_time(self):
        """Get the amount of time shown on the x-axis of the bottom two graphs.

        :return: The amount of time
        :rtype: Float
        """
        return self.plot_time

    def set_plot_time(self,plot_time):
        """Set the amount of time shown on the x-axis of the bottom two graphs.

        :param plot_time: The amount of time
        :type plot_time: Float
        :return: Nothing
        :rtype: None
        """
        self.plot_time = plot_time
        self.restart_required = True

    def get_demodulate_plot_points(self):
        """Get the number of points plotted in the bottom two plots. Setting this to a to high value will make things slow, settings this to a too high value will cause aliasing.

        :return: The number of points
        :rtype: Float
        """
        return self.demodulate_plot_points

    def get_integration_time(self):
        """Get the time over which the demodulation signals will be avaraged to compute the offset displayed in the interface and the offset obtain using :func:`TiePieLCR_api.TiePieLCR_api.get_impedance`.

        :return: The amount of time
        :rtype: Float
        """
        return self.integration_time

    def set_integration_time(self,integration_time):
        """Set the time over which the demodulation signals will be avaraged to compute the offset displayed in the interface and the offset obtain using :func:`TiePieLCR_api.TiePieLCR_api.get_impedance`.

        :param integration_time: The amount of time
        :type integration_time: Float
        :return: Nothing
        :rtype: None
        """
        self.integration_time = integration_time
        self.restart_required = True


    def get_settings_dict(self,filename):
        """Get the settings in the object formatted as a dictionary object.

        :return: A dict with the settings in this object
        :rtype: Dictionary
        """

        self.save_settings(filename)
        self.load_settings(filename,self.inst)

        self.load_calibration()
        self.settings_dict['calibration'] = self.calibration.get_calibration_dict('calibrations/%s.yaml'%(self.frontend_number))
        return self.settings_dict
    
    def get_demodulation_time_vector(self,blocks):
        """Calculate the time vector for a certain number of block of demodulation data.

        :param blocks: The number of blocks of scope data for which a time vector should be calcualted
        :type blocks: Int
        :return: The calculated time vector
        :rtype: Numpy vector
        """
        n = blocks*self.get_final_output_block_size()
        tblock = 1.0/self.get_update_frequency()
        dt = tblock/self.get_final_output_block_size()
        return (np.arange(0,n)+self.get_final_output_block_size()/2)*dt
        
    def get_offset_time_vector(self,blocks):
        """Calculate the time vector for a certain number of block of offset data.

        :param blocks: The number of blocks of scope data for which a time vector should be calcualted
        :type blocks: Int
        :return: The calculated time vector
        :rtype: Numpy vector
        """
        tblock = 1.0/self.get_update_frequency()
        dt_offset = tblock/self.get_final_offset_block_size()
        n_offset = blocks*self.get_final_offset_block_size()
        return (np.arange(0,n_offset)+self.get_final_offset_block_size()/2)*dt_offset

    def get_multisine_vector(self):
        """Get the multisine signal that the AWG in the TiePieLCR will be set to. 

        :return: The calculated multisine signal
        :rtype: Numpy vector
        """
        dt = 1.0/self.get_gen_sample_freq()
        t = np.arange(0,self.get_gen_samples()*dt-dt/2,dt)
        y = np.zeros(self.get_gen_samples())
        for i1 in range(self.get_number_of_multisine_freqs()):
            y = y + self.gen_weights[i1]*np.sin(2*np.pi*self.gen_freqs[i1]*t+self.gen_phases[i1])

        y = y/np.max(np.abs(y))
        # y = y + self.get_gen_offset()
        

        data_array = array('f')

        data_array.extend(np.asarray(y, dtype='f').tolist())
        return data_array

    def get_multisine_crest_factor(self):
        dt = 1.0/self.get_gen_sample_freq()
        t = np.arange(0,self.get_gen_samples()*dt-dt/2,dt)
        y = np.zeros(self.get_gen_samples())
        for i1 in range(self.get_number_of_multisine_freqs()):
            y = y + self.gen_weights[i1]*np.sin(2*np.pi*self.gen_freqs[i1]*t+self.gen_phases[i1])

        crest = np.max(np.abs(y))/np.mean(y**2)**0.5
        return crest

    def reset(self):
        """When the settings are loaded in the TiePieLCR parts interface of the interface might be reset. By running this function before any changes are made to the settings object, only the parts that really need to be reset are reset.

        :return: Nothing
        :rtype: None
        """
        self.restart_required = False
        self.reference_update_required = False
        self.gain_update_required = False
        self.multisine_update_required = False
        self.gen_restart_required = False
        self.base_vector_update_required = False
        self.gen_offset_update_required = False
        self.gen_amplitude_update_required = False
        self.dem_freq_update_required = False

    

    def set_minimum_plot_frequency(self,frequency):
        """Set the minimum frequency that is shown in the frequency plots on the top right.

        :return: True if smaller as the current maximum frequency, False otherwise.
        :rtype: Boolean
        """
        if frequency < self.fmax_plot:
            self.fmin_plot = frequency
            return True
        else:
            return False
    
    def set_maximum_plot_frequency(self,frequency):
        """Set the minimum frequency that is shown in the frequency plots on the top right.

        :return: True if larger as the current minimum frequency, False otherwise.
        :rtype: Boolean
        """
        if frequency > self.fmin_plot:
            self.fmax_plot = frequency
            return True
        else:
            return False

    def get_sample_frequency(self):
        """Get the sample frequency at which the scope is running. A higher value results in less noise, however it might be reduced if the LCR stop and gives an error complaining that your PC is not fast enough.

        :return: The sample frequency
        :rtype: Float
        """
        return self.sample_frequency    

    def set_sample_frequency(self,value):
        """Set the sampling frequency of the scope. A higher value results in less noise, however it might be reduced if the LCR stop and gives an error complaining that your PC is not fast enough.
        
        The sampling frequency should be smaller as 6250000 Hz and be an integer multiple of the number of sub blocks.

        :return: A dictionary that contains the keyword 'error' when an error occured and is empty otherwise
        :rtype: Dict
        """
        actual_value = self.fs_list[value]
        package = {}
        new_block_size = int(actual_value/self.get_update_frequency())
        if actual_value > self.max_sample_frequency:
            package['error'] = b'error: sample frequency above maximum sample frequency'
            print("invalid sample freq")
            return package
        elif not new_block_size%self.get_sub_blocks() == 0:
            package['error'] = b'error: block size should be an integer multiple of sub_blocks'
            print("block_size_error while setting sample frequency")
            return package
        else:
            self.fs_setting = value
            self.sample_frequency = actual_value
            #self.calculate_output_parameters()
            self.restart_required = True
            self.base_vector_update_required = True
            return package

    def get_update_frequency(self):
        """Get how often data is retrieved from the scope, the gui is updated and how often the value obtained using :func:`TiePieLCR_api.TiePieLCR_api.get_impedance` is updated. Only reduce this if your computer is really really slow..

        :return: The update frequency
        :rtype: Float
        """
        return self.update_freq_list[self.update_freq_setting]


    def set_update_freq(self,update_freq_index):
        """Set how often data is retrieved from the scope, the gui is updated and how often the value obtained using :func:`TiePieLCR_api.TiePieLCR_api.get_impedance` is updated. Only reduce this if your computer is really really slow..

        The update sample frequency divided by the update frequency should be an integer multiple of the number of sub blocks.

        :param update_freq_index: Frequency in the list obtained through :func:`TiePieLCR_settings.TiePieLCR_settings.get_update_freq_list` that should be used
        :type update_freq_index: Int
        :return: A dictionary that contains the keyword 'error' with an error when an error occured and is empty otherwise
        :rtype: Dict
        """
        new_block_size = self.sample_frequency/self.update_freq_list[update_freq_index]
        package ={}
        if not new_block_size%self.get_sub_blocks() == 0:
            package['error'] = b'error: block size should be an integer multiple of sub_blocks'
            print("block_size_error while setting update frequency")
            return package
        else:
            self.update_freq_setting = update_freq_index
            self.restart_required = True
            self.base_vector_update_required = True
            #self.calculate_output_parameters()
            return package
    
    

    def set_demodulation_bandwidth(self,bandwidth):
        """Set the bandwidth with which the current and the voltage are demodulated and how fast the bottom two graphs in the interface will respond.

        :param bandwidth: The bandwidth
        :type bandwidth: Float
        :return: A dictionary that contains the keyword 'error' with an error when an error occured and is empty otherwise
        :rtype: Dict
        """
        package = {}
        if bandwidth >= min(self.dem_freqs):  
            package['error'] = b'error: bandwidht must be larger as the minimum frequency'
            return package
        if bandwidth <= 0:
            package['error'] = b'error: bandwidht must be positive'
            return package
        self.bandwidth = bandwidth
        self.restart_required = True
        self.base_vector_update_required = True
        return package


    

    def get_sub_blocks_list(self):
        """ get a list of options to which the sub-blocks can be set.

        :return: A list with the possible sub-block settings
        :rtype: List of Ints
        """
        return self.sub_blocks_list

    def get_sub_block_size(self):
        """Get the size of the blocks on which the fft is performed. Since the time it takes to do an fft increases with a power of 1.4 with the amount of points, a higher number of sub-blocks will reduce the computation load. A higher number sub-blocks also moves the frequency at which noise at the edge of each block will appear. A higher number of sub-blocks makes very lower frequency measurements more computationally intensive though.

        :return: The number of samples per sub-block
        :rtype: Int
        """
        return int(self.get_block_size()/self.get_sub_blocks())

    #the number of sub blocks per second
    def get_sub_block_freq(self):
        """The number of sub blocks per second that need to be processed. This equavalent to the number of FFT's that are being taken per second.

        :return: The number of sub-blocks per second
        :rtype: Int
        """
        return self.get_update_frequency()*self.get_sub_blocks()

    #the number of subblocks one block of data from the scope is defided into
    def get_sub_blocks(self):
        """Get in how many blocks the data retrieved from the scope is split up before doing the fft. Since the time it takes to do an fft increases with a power of 1.4 with the amount of points, a higher number of sub-blocks will reduce the computation load. A higher number sub-blocks also moves the frequency at which noise at the edge of each block will appear. A higher number of sub-blocks makes very lower frequency measurements more computationally intensive though.

        :return: The number of sub-blocks
        :rtype: Int
        """
        return self.sub_blocks_list[self.sub_blocks_setting]

    

    def set_sub_blocks(self,sub_block_index):
        """Sets in how many blocks the data retrieved from the scope is split up before doing the fft. Since the time it takes to do an fft increases with a power of 1.4 with the amount of points, a higher number of sub-blocks will reduce the computation load. A higher number sub-blocks also moves the frequency at which noise at the edge of each block will appear. A higher number of sub-blocks makes very lower frequency measurements more computationally intensive though.

        :param sub_block_index: Number in the list obtained through :func:`TiePieLCR_settings.TiePieLCR_settings.get_sub_blocks_list` that should be used
        :type sub_block_index: Int
        :return: A dictionary that contains the keyword 'error' with an error when an error occured and is empty otherwise
        :rtype: Dict
        """
        package ={}
        if not self.get_block_size()%self.sub_blocks_list[sub_block_index] == 0:
            package['error'] = b'error: block size should be an integer multiple of sub_blocks'
            print("block_size_error while stting sub blocks")
            return package
        else:
            self.sub_blocks_setting = sub_block_index
            self.restart_required = True
            self.base_vector_update_required = True
            #self.calculate_output_parameters()
            return package

    def get_df(self):
        """Get number of frequencies in one bin of the fft

        :return: The number of frequencies in one bin
        :rtype: Float
        """
        return float(self.sample_frequency)/float(self.get_sub_block_size())/2

    def get_block_size(self):
        """Get number of samples in each block of data that will be retrieved from the scope

        :return: The number of samples in one block
        :rtype: Int
        """
        return self.sample_frequency/self.get_update_frequency()
    
    def get_output_oversample_ratio(self):
        """Get the number of times the output signal will be oversamples. The sample frequency fill be oversample ratio times 2 times the bandwidth.

        :return: The oversampling ratio of the output
        :rtype: Int
        """
        return self.output_oversample_ratio

    def get_output_sub_block_size(self):
        """Get number of samples of impedance samples that will be calculated during each sub-block using the ffts, before the impedance signal is downsampled.
        
        
        This currently is calculated incorrectly and will probably cause the gui to crash if the demodulation frequency is set too high

        :return: The number of impedance samples per sub-block
        :rtype: Int
        """
        desired_n = int(self.bandwidth/self.get_df()*self.output_oversample_ratio)
        if desired_n < self.side_lob_n:
            return int(self.side_lob_n)*2
        else:
            return desired_n*2
    
    #the size of one block after processing
    def get_output_block_size(self):
        """Get number of impedance samples that will be calculated during each block using the ffts, before the impedance signal is downsampled.
        
        This currently is calculated incorrectly and will probably cause the gui to crash if the demodulation frequency is set too high

        :return: The number of impedance samples per block
        :rtype: Int
        """
        return int(self.get_sub_blocks()*self.get_output_sub_block_size())

    #the samples frequency after processing
    def get_output_sample_freq(self):
        """Get the sample frequency of the impedance samples that will be calculated using the ffts, before the impedance signal is downsampled.

        :return: Number of impedance samples per second
        :rtype: Float
        """
        return self.get_output_block_size()*self.get_update_frequency()
    
    def get_plot_blocks(self):
        """Get how many blocks of data should be used to plot impedance data for a period equal to the plot time.

        :return: Number of blocks
        :rtype: Int
        """
        return int(self.get_update_frequency()*self.plot_time)
    
    def get_offset_sub_block_size(self):
        """Get number of offset samples that will be calculated during each sub-block using the ffts, before the impedance signal is downsampled. 

        :return: The number of offset samples per sub-block
        :rtype: Int
        """
        return 2*(int(self.get_offset_bandwidth()/self.get_df())+10)

    def get_offset_block_size(self):
        """Get number of offset samples that will be calculated during each block using the ffts, before the impedance signal is downsampled. 

        :return: The number of offset samples per sub-block
        :rtype: Int
        """
        return self.get_offset_sub_block_size()*self.get_sub_blocks()

    def get_offset_sample_frequency(self):
        """Get the sample frequency of the offset samples that will be calculated using the ffts, before the offset signal is downsampled.

        :return: Number of offset samples per second
        :rtype: Float
        """
        return self.get_offset_block_size()*self.get_update_frequency()

    def get_output_downsampling_rate(self):
        """Get the factor by which the impedance signal calculated using the ffts will be downsampled.

        :return: The downsampling ratio
        :rtype: Int
        """
        Wn = self.bandwidth/self.get_update_frequency()/self.get_output_block_size()*2
        downsampling_rate = self.calculate_downsapling_rate(self.get_output_block_size(), Wn)
        return downsampling_rate

    def get_final_output_block_size(self):
        """Get the number of impedance samples that will result from each block after downsampling. 

        :return: The number of samples
        :rtype: Int
        """
        return int(self.get_output_block_size()/self.get_output_downsampling_rate())

    def get_offset_downsampling_rate(self):
        """Get the factor by which the offset signal calculated using the ffts will be downsampled.

        :return: The downsampling ratio
        :rtype: Int
        """
        Wn = self.offset_bandwidth/self.get_update_frequency()/self.get_offset_block_size()*2
        return self.calculate_downsapling_rate(self.get_offset_block_size(), Wn)
    
    def get_final_offset_block_size(self):
        """Get the number of offset samples that will result from each block after downsampling. 

        :return: The number of samples
        :rtype: Int
        """
        return int(self.get_offset_block_size()/self.get_offset_downsampling_rate())        

    def get_fft_sensitivity(self):
        """Calculate the minimum width of a peak in the fft

        :return: The minimum widht in Hertz
        :rtype: Float
        """
        return self.side_lob_n*self.get_df()

    def get_number_of_multisine_freqs(self):
        """The of frequencies in the multisine

        :return: The number of frequencies
        :rtype: Int
        """
        return len(self.gen_freqs)

    def get_number_of_demodulation_freqs(self):
        """The of frequencies in the multisine

        :return: The number of frequencies
        :rtype: Int
        """
        return len(self.dem_freqs)

    @staticmethod
    def calculate_downsapling_rate(n,Wn):
        """Calculate the maximum possible downsampling rate

        :return: The downsampling rate
        :rtype: Complex double
        """
        desired_downsampling_rate = int(1/Wn/5)
        for downsampling_rate in range(desired_downsampling_rate,0,-1):
            if n%downsampling_rate == 0:
                final_downsampling_rate = downsampling_rate
                break
        if desired_downsampling_rate <= 1:
            final_downsampling_rate = 1
        return final_downsampling_rate
    
    def get_f_fun(self):
        """Calculate frequency with which the multisine will repeat itself.

        :return: The fundamental frequency
        :rtype: float
        """
        if self.get_number_of_multisine_freqs() == 0:
            return 0

        return self.gcd(self.gen_freqs)

    def get_f_max(self):
        """The maximum frequency in the multisine

        :return: The maximum frequency
        :rtype: float
        """
        return max(self.gen_freqs) 
    
    def get_f_min(self):
        """The minimum frequency in the multisine

        :return: The minimum frequency
        :rtype: float
        """
        return min(self.gen_freqs)

    def get_gen_samples(self):
        """Calculate the number of samples that should be used for the multisine

        :return: The number of samples
        :rtype: Int
        """
        gen_samples, gen_sample_freq = self.calculate_gen_data_size()
        #print("gen_samples: "+str(gen_samples))
        return gen_samples
    
    def get_gen_sample_freq(self):
        """Calculate the sample frequency that should be used for the multisine

        :return: The sample frequency
        :rtype: Float
        """
        gen_samples,gen_sample_freq = self.calculate_gen_data_size()
        #print("gen_sample_freq: "+str(gen_sample_freq))
        return gen_sample_freq
    
    def calculate_gen_data_size(self):
        f_fun = self.get_f_fun()
        if f_fun ==0:
            return int(self.gen_samples_min),self.gen_sample_freq_max
        required_samples = self.gen_sample_freq_max/f_fun
        if required_samples < self.gen_samples_max:
            if required_samples < self.gen_samples_min:
                gen_sample_freq = self.gen_sample_freq_max
                periods = int(f_fun/(self.gen_sample_freq_max/self.gen_samples_min))
                gen_samples = int(self.gen_sample_freq_max/f_fun*periods)
            else:
                gen_sample_freq = self.gen_sample_freq_max
                gen_samples = int(required_samples)
        else:
            gen_samples = int(self.gen_samples_max)
            gen_sample_freq = self.gen_samples_max*f_fun
        return gen_samples, gen_sample_freq

    def set_demodulation_params(self,freq_list,tiepie_list):
        package = {}
        if len(freq_list) ==0:
            package['warning'] = b'warning: no demodulate frequencies found'
            return package
        for tiepies in tiepie_list:
            if tiepies < 1:
                package['error'] = b'Tiepie# should be the number of the tiepie that has the reference for this frequency. This number should be larger than zero'
                return package
        if min(freq_list) <= self.bandwidth:
            package['error'] = b'error: cannot set a demodulation frequency lower that the demodulation bandwidth'
            return package
        if max(freq_list) >= self.sample_frequency/2:
            package['error'] = b'error: cannot set a demodulation frequency equal or higher as the half the sample rate'
            return package
        
        self.dem_tiepies = np.array(tiepie_list)

        #make array of the generator frequencies
        if len(np.array(freq_list))!=len(self.dem_freqs):
            self.restart_required = True
            print("requesting restart")
            self.base_vector_update_required = True

        self.dem_freqs = np.array(freq_list)

        self.dem_freq_update_required = True
        
        return package

    def set_multisine(self,freq_list,weight_list):
        """Sets the frequencies and their weights in the multisine that is used as an excitation signal. The weights are defined relative to the maximum amplitude of the signal.

        :param freq_list: A list of frequencies to be used in the multisine
        :type freq_list: List of floats
        :param weight_list: A list of weight to be used in the multisine. The weights are relative to the maximum amplitude of the signal as defined by :func:`TiePieLCR_settings.TiePieLCR_settings.set_amplitude`.
        :type weight_list: List of weights
        :return: The sample frequency
        :rtype: Float
        """
        package = {}
        if len(freq_list) > 0:
            if max(freq_list) >= self.gen_sample_freq_max/10:
                package['error'] = b'error: cannot set a generator frequency equal or higher as one tenth the maximum sample rate of the generator'
                return package

        #store weights
        self.gen_weights = np.array(weight_list)
        #normalise weights
        self.gen_weights = np.around(self.gen_weights,decimals=2)

        self.gen_freqs = np.array(freq_list)

        if len(self.gen_phases) != len(freq_list):
            self.gen_phases = np.zeros((len(freq_list)))        

        self.multisine_update_required = True

        if self.dem_is_gen and len(np.array(freq_list))!=len(self.dem_freqs):
            self.gen_restart_required = True
            print("requesting generator restart")
        
        return package

    @staticmethod
    def lcm(L):
        lcm = L[0]
        for i in L[1:]:
            lcm = int(lcm*i/math.gcd(lcm, i))
        return lcm
    
    @staticmethod
    def gcd(L):
        decimals = 3
        L_int = []
        for item in L:
            L_int.append(int(item*10**decimals))
        gcd = L_int[0]
        for item in L_int[1:]:
             gcd = math.gcd(gcd,item)
        return gcd/10**decimals


    def set_amplitude(self,amplitude):
        """Sets the maximum amplitude of the multisine that is used as an excitation signal.

        :param amplitude: The maximum amplitude
        :type amplitude: Float
        :return: False if the total excitation signals get's too large, True otherwise
        :rtype: Boolean
        """
        if amplitude+np.abs(self.gen_offset)>self.Vmax:
            return False
        else:
            self.gen_amplitude = amplitude
            self.gen_amplitude_update_required = True
            return True

    def set_offset(self,offset):
        """Sets the offset of the multisine that is used as an excitation signal.

        :param amplitude: The offset
        :type amplitude: Float
        :return: False if the total excitation signals get's too large, True otherwise
        :rtype: Boolean
        """
        if self.gen_amplitude+np.abs(offset)>self.Vmax:
            return False
        else:
            self.gen_offset = offset
            self.gen_offset_update_required = True
            #self.gen_restart_required = True
            return True

    def set_frontend_number(self,frontend_number):
        self.frontend_number = frontend_number
        self.load_calibration()

    def get_frontend_number(self):
        return self.frontend_number

    def load_calibration(self):
        self.calibration = TiePieLCR_calibration()
        new_calibration = self.calibration
        if new_calibration.load_calibration('calibrations/%s.yaml'%(self.frontend_number)):
            self.calibration = new_calibration
            return True
        else:
            self.calibration = TiePieLCR_calibration()
            return False

    def load_settings(self, filename,instance):
        """Loads a settings file and applies the settings to this object

        :param filename: The filename
        :type filename: String
        :return: True if the file exists, False otherwise
        :rtype: Boolean
        """
        try:
            with open(filename, 'r', encoding='utf-8') as stream:
                settings = yaml.safe_load(stream)
        except:
            self.dem_tiepies = [instance+1]
            return False

        self.settings_dict = settings
    

        if 'fs_name_list' in settings:
            self.fs_name_list = settings['fs_name_list']
        if 'fs_list' in settings:
            self.fs_list = settings['fs_list']
        if 'fs_setting' in settings:
            self.fs_setting = settings['fs_setting']
        self.set_sample_frequency(self.fs_setting)

        if 'update_freq_name_list' in settings:
            self.update_freq_name_list = settings['update_freq_name_list']
        if 'update_freq_list' in settings:
            self.update_freq_list = settings['update_freq_list']
        if 'update_freq_setting' in settings:
            self.update_freq_setting = settings['update_freq_setting']

        if 'sub_blocks_name_list' in settings:
            self.sub_blocks_name_list = settings['sub_blocks_name_list']
        if 'sub_blocks_list' in settings:
            self.sub_blocks_list = settings['sub_blocks_list']
        if 'sub_blocks_setting' in settings:
            self.sub_blocks_setting = settings['sub_blocks_setting']

        if 'scope_ranges' in settings:
            self.scope_ranges = settings['scope_ranges']
        if 'scope_range_name_list' in settings:
            self.scope_range_name_list = settings['scope_range_name_list']

        if 'scope_couplings' in settings:
            self.scope_couplings = settings['scope_couplings']
        if 'scope_coupling_name_list' in settings:
            self.scope_coupling_name_list = settings['scope_coupling_name_list']
        
        if 'gain' in settings:
            self.gain_setting = settings['gain']

        if 'reference' in settings:
            self.reference_setting = settings['reference']
    
        if 'gen_amplitude' in settings:
             self.gen_amplitude = settings['gen_amplitude']
        if 'gen_offset' in settings:
             self.gen_offset = settings['gen_offset']
        if 'gen_freqs' in settings:
             self.gen_freqs = np.array(settings['gen_freqs'])
        if 'gen_weights' in settings:
             self.gen_weights = np.array(settings['gen_weights'])
        if 'gen_phases' in settings:
             self.gen_phases = np.array(settings['gen_phases'])

        if 'dem_freqs' in settings:
             self.dem_freqs = np.array(settings['dem_freqs'])
        if 'dem_tiepies' in settings:
             self.dem_tiepies = np.array(settings['dem_tiepies'])
             
        if 'dem_is_gen' in settings:
             self.dem_is_gen = bool(settings['dem_is_gen'])

        if 'plot_points' in settings:
            self.plot_points= settings['plot_points']
        if 'plot_periods' in settings:
            self.plot_periods = settings['plot_periods']
        if 'fft_plot_points' in settings:
            self.fft_plot_points = settings['fft_plot_points']
        if 'fmin_plot' in settings:
            self.fmin_plot = settings['fmin_plot']
        if 'fmax_plot' in settings:
            self.fmax_plot = settings['fmax_plot']


        if 'bandwidth' in settings:
            self.bandwidth = settings['bandwidth']
        if 'plot_time' in settings:
            self.plot_time = settings['plot_time']
        if 'integration_time' in settings:
            self.integration_time = settings['integration_time']    
        if 'impedance_format' in settings:
            self.impedance_format = settings['impedance_format']
        if 'impedance_formats' in settings:
            self.impedance_formats = settings['impedance_formats']    
        if 'impedance_format_unit1' in settings:
            self.impedance_format_unit1 = settings['impedance_format_unit1']
        if 'impedance_format_unit2' in settings:
            self.impedance_format_unit2 = settings['impedance_format_unit2']
        if 'impedance_format_label1' in settings:
            self.impedance_format_label1 = settings['impedance_format_label1']
        if 'impedance_format_label2' in settings:
            self.impedance_format_label2 = settings['impedance_format_label2']
        if 'offset_bandwidth' in settings:
            self.offset_bandwidth = settings['offset_bandwidth']
        if 'offset_integration_time' in settings:
            self.offset_integration_time = settings['offset_integration_time']

        if 'real_time_mode' in settings:
            self.real_time_mode = settings['real_time_mode']
        if 'no_frontend' in settings:
            self.no_frontend = settings['no_frontend']
        if 'frontend_number' in settings:
            self.frontend_number = settings['frontend_number']

        # self.load_calibration()
        # self.settings_dict['calibration'] = self.calibration.get_calibration_dict('calibrations/%s.yaml'%(self.frontend_number))
        self.set_multisine(self.gen_freqs,self.gen_weights)
        return True

    def save_settings(self,filename):
        """Saves a settings file and with the settings of this object

        :param filename: The filename
        :type filename: String
        :return: Nothing
        :rtype: none
        """
        settings = {}

        settings['version'] = self.version

        settings['fs_name_list'] = self.fs_name_list
        settings['fs_list'] = self.fs_list
        settings['fs_setting'] = self.fs_setting

        settings['update_freq_name_list'] = self.update_freq_name_list
        settings['update_freq_list'] = self.update_freq_list
        settings['update_freq_setting'] = self.update_freq_setting

        settings['sub_blocks_name_list'] = self.sub_blocks_name_list
        settings['sub_blocks_list'] = self.sub_blocks_list
        settings['sub_blocks_setting'] = self.sub_blocks_setting

        settings['scope_ranges'] = self.scope_ranges
        settings['scope_range_list'] = self.scope_range_list
        settings['scope_range_name_list'] = self.scope_range_name_list

        settings['scope_couplings'] = self.scope_couplings
        settings['scope_coupling_name_list'] = self.scope_coupling_name_list

        settings['gain'] = self.gain_setting


        settings['reference'] = self.reference_setting
 


        settings['gen_amplitude'] = self.gen_amplitude
        settings['gen_offset'] = self.gen_offset
        settings['gen_freqs'] = np.array(self.gen_freqs).tolist()
        settings['gen_weights'] = np.array(self.gen_weights).tolist()
        settings['gen_phases'] = np.array(self.gen_phases).tolist()

        settings['dem_freqs'] = np.array(self.dem_freqs).tolist()
        settings['dem_tiepies'] = np.array(self.dem_tiepies).tolist()
        settings['dem_is_gen'] = bool(self.dem_is_gen)
        

        settings['plot_points'] = self.plot_points
        settings['fft_plot_points'] = self.fft_plot_points
        settings['plot_periods'] = self.plot_periods
        settings['fmin_plot'] = self.fmin_plot
        settings['fmax_plot'] = self.fmax_plot

        settings['bandwidth'] = self.bandwidth
        settings['plot_time'] = self.plot_time
        settings['integration_time'] = self.integration_time
        settings['impedance_format'] = self.impedance_format
        settings['impedance_formats'] = self.impedance_formats
        settings['impedance_format_unit1'] = self.impedance_format_unit1
        settings['impedance_format_unit2'] = self.impedance_format_unit2
        settings['impedance_format_label1'] = self.impedance_format_label1
        settings['impedance_format_label2'] = self.impedance_format_label2

        settings['offset_bandwidth'] = self.offset_bandwidth
        settings['offset_integration_time'] = self.offset_integration_time

        settings['real_time_mode'] = self.real_time_mode
        settings['serial_numbers'] = self.serial_numbers
        settings['no_frontend'] = self.no_frontend
        settings['frontend_number'] = self.frontend_number

        with io.open(filename, 'w', encoding='utf8') as outfile:
                yaml.dump(settings, outfile, default_flow_style=False, allow_unicode=True)