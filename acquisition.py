"""
.. module:: acquisition
   :synopsis: This class contains code that runs on a seperate core and that does the actual acquisition and processing of the data
.. moduleauthor:: Martijn Schouten <github.com/martijnschouten>
"""

from TiePieLCR import TiePieLCR
import threading
import time
from multiprocessing import Value, Array, freeze_support


class acquisition:
    package_list = []
    fft_package_list = []
    def __init__(self,LCR_settings_queue,displayed_error,shared_state,plot_data_queue,stored_data_queue,stored_data_requested):
        print("starting aquisition class")
        self.settings_queue = LCR_settings_queue
        self.displayed_error = displayed_error
        self.shared_state = shared_state
        self.plot_data_queue = plot_data_queue
        self.LCR_settings_queue = LCR_settings_queue
        self.all_settings = LCR_settings_queue.get()
        self.lockins = len(self.all_settings)
        for i1 in range(self.lockins):#for some reason calibration is not synced the first time
            self.all_settings[i1].load_calibration()
        self.stored_data_queue = stored_data_queue
        self.stored_data_requested = stored_data_requested
        self.build_tiepie_list()
        print("initialised objects")

        self.thread = threading.Thread(target=self.data_aquisition_daemon, args=())
        self.thread.start()
        self.thread = threading.Thread(target=self.fft_daemon, args=())
        self.thread.start()
        self.thread = threading.Thread(target=self.processing_daemon, args=())
        self.thread.start()

        print("opened LCR object")

    def build_tiepie_list(self):
        self.TiepieLCR = []
        for i1 in range(self.lockins):
            self.TiepieLCR.append(TiePieLCR(self.all_settings,i1))

    def data_aquisition_daemon(self):
        print('started data acquisition daemon')
        while(1):
            while not self.settings_queue.empty():
                self.all_settings = self.settings_queue.get_nowait()

                if self.lockins != len(self.all_settings):
                    if self.shared_state.value == 2:
                        self.TiepieLCR[0].stop_measurement()
                        self.shared_state.value = 0
                        time.sleep(0.5)
                    self.lockins = len(self.all_settings)
                    self.build_tiepie_list()

                old_state = self.shared_state.value
                self.shared_state.value = 5
                for i1 in range(self.lockins):
                    if not self.TiepieLCR[i1].set_settings(self.all_settings,i1,self.TiepieLCR[0]):
                        print('Could not set settings.')
                self.shared_state.value = old_state
            if self.shared_state.value == 0 or self.shared_state.value == -1:
                time.sleep(0.001)
            elif self.shared_state.value == 1:
                while len(self.package_list) > 0:
                    self.package_list.pop(0)
                package =  self.TiepieLCR[0].start_measurement(self.all_settings)
                for i1 in range(0,self.lockins):
                    self.TiepieLCR[i1].reset_buffers()
                if 'error' in package:
                    self.shared_state.value = 4
                    self.displayed_error.value = package['error']
                else:
                    self.shared_state.value = 2
                    print("started LCR")
                    #tic=time.time()
            elif self.shared_state.value == 2:
                #print("starting up took: "+str(time.time()-tic))
                package = self.TiepieLCR[0].get_data(self.all_settings)
                
                

                if 'error' in package:
                    self.displayed_error.value = package['error']
                    self.TiepieLCR[0].stop_measurement()
                    self.shared_state.value = 0
                    continue
                if 'timeout' in package:
                    self.TiepieLCR[0].stop_measurement()
                    self.shared_state.value = 0
                    continue
                elif 'gone' in package:
                    self.shared_state.value = 0
                    self.displayed_error.value = package['gone']
                    continue
                elif 'warning' in package:
                    self.displayed_error.value = package['warning']
                    continue

                self.package_list.append(package)                

            elif self.shared_state.value == 3:
                self.TiepieLCR[0].stop_measurement()
                self.shared_state.value = 0
            elif self.shared_state.value == 4:
                #wait for the subprocess to be terminated
                time.sleep(0.1)
            else:
                print("entered unkown state number " + str(self.shared_state.value))
            if self.shared_state.value >= 0:
                if self.stored_data_requested.value == 1:
                    storage_package = []
                    error = False
                    for i1 in range(self.lockins):
                        storage_package.append(self.TiepieLCR[i1].get_stored_data())
                        if 'error' in storage_package[i1]:
                            error = True
                    if error:
                        storage_package['error'] = True
                    self.stored_data_queue.put(storage_package)
                    self.stored_data_requested.value = 0

    def fft_daemon(self):
        tic = time.time()
        while(1):
            #print("restarted processing loop")
            if self.shared_state.value == 2:
                if self.all_settings[0].real_time_mode:
                    if len(self.package_list) > 1:
                        while len(self.package_list) > 1:
                            self.package_list.pop(0)
                        self.displayed_error.value = b'warning: dropping packages because realtime mode is enable and cannot keep up processing'
                        print("dropped package")
                else:

                    if len(self.package_list) > 3:
                        self.displayed_error.value = b'warning having trouble keeping up processing: try reducing sampling rate or getting a faster computer'
                        print("can't keep up processing data. Queue longer than "+ str(len(self.package_list)))
                        
                while(len(self.package_list) == 0):
                    time.sleep(0.001)

                data_package = self.package_list[0]
                #print(data_package['signal'][0])
                self.package_list.pop(0)

                #tic = time.time()
                fft_package = self.TiepieLCR[0].do_the_ffts(data_package)
                #print("fft time: " + str(time.time()-tic))

                if 'error' in fft_package:
                    self.displayed_error.value = fft_package['error']
                    self.TiepieLCR[0].stop_measurement()
                    self.shared_state.value = 0
                    time.sleep(0.5)
                    continue
                elif 'warning' in fft_package:
                    self.displayed_error.value = fft_package['warning']
                    continue

                self.fft_package_list.append(fft_package)
            else:
                time.sleep(0.001)

    def processing_daemon(self):
        
        
        while(1):
            #print("restarted processing loop")
            if self.shared_state.value == 2:                
                if self.all_settings[0].real_time_mode:
                    if len(self.fft_package_list) > 1:
                        while len(self.fft_package_list) > 1:
                            self.package_list.pop(0)
                        self.displayed_error.value = b'warning: dropping fft packages because realtime mode is enable and cannot keep up processing'
                        print("dropped fft package")
                else:
                    if len(self.fft_package_list) > 3:
                        self.displayed_error.value = b'warning: having trouble keeping up fft processing: try reducing sampling rate or getting a faster computer'
                        print("can't keep up fft processing data. Queue longer than "+ str(len(self.fft_package_list)))
                
                while(len(self.fft_package_list) == 0):
                    time.sleep(0.001)
                
                fft_package = self.fft_package_list[0]
                self.fft_package_list.pop(0)

                #tic = time.time()
                plot_package = []
                for i1 in range(self.lockins):  
                    plot_package.append(self.TiepieLCR[i1].process_data(fft_package))
                #print("processing time: " + str(time.time()-tic))

                if self.plot_data_queue.empty() == False:
                    #self.displayed_error.value = b'cannot keep up plotting: try reducing plot frequency'
                    while not self.plot_data_queue.empty():
                        try:
                            self.plot_data_queue.get_nowait()
                        except: 
                            print('cant even emtpy queue')
                    #print('cannot keep up plotting')
                
                issue = False
                for i1 in range(self.lockins):
                    if 'error' in plot_package[i1]:
                        self.displayed_error.value = plot_package[i1]['error']
                        self.TiepieLCR[0].stop_measurement()
                        self.shared_state.value = 0
                        time.sleep(0.5)
                        print('error in plot package')
                        issue = True
                    elif 'warning' in plot_package[i1]:
                        print(plot_package[i1]['warning'])
                        issue = True

                
                if not issue:
                    self.plot_data_queue.put(plot_package)
                
                #self.new_data = False
            else:
                time.sleep(0.001) 