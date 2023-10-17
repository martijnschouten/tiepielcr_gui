"""
.. module:: MainWindow
   :synopsis: This class contains the code for the part of the GUI that is not specific
.. moduleauthor:: Martijn Schouten <github.com/martijnschouten>
"""

import sys  # We need sys so that we can pass argv to QApplication
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
import threading
import time
from multiprocessing import Process, Queue, Value, Array, freeze_support
import scipy.io as sio
import numpy as np
import socket
from impedance import impedance
import pickle
import os
from acquisition import acquisition
from lockin_tab import lockin_tab
from TiePieLCR import gpu_available
from TiePieLCR_settings import TiePieLCR_settings
import resource

class MainWindow(QtWidgets.QMainWindow):
    state = -1
    updating_gui = False
    lockins = 1
    reference_range = 0
    new_data = False

    plot_data_queue = None
    settings_queue = Queue()
    displayed_error = Array('c',  300)

    settings_filename = 'settings'
    settings_folder = 'settings/'

    updating_common_gui = False
    old_error = ''

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.state = -1

        #Load the UI Page
        uic.loadUi('interface.ui', self)

        if gpu_available == True:
            self.GPU_en_value.setText('True')
        else:
            self.GPU_en_value.setText('False') 

        self.lockins = 0
        while True:
            filename_string = self.settings_folder+self.settings_filename+'_'+str(self.lockins+1)+'.yaml'
            print(filename_string)
            if os.path.isfile(filename_string):
                self.lockins = self.lockins + 1
            else:
                break

        if self.lockins == 0:
            print('No settings file found')
            self.lockins = 1

        self.lockin_spinner.setValue(self.lockins)
        self.lockin_widgets = []
        for i1 in range(self.lockins):
            self.lockin_widgets.append(lockin_tab(self,i1))
            self.tabWidget.insertTab(i1,self.lockin_widgets[i1], 'lockin '+str(i1+1))
        self.tabWidget.setCurrentIndex(0)
        
        #this needs to be here otherwise the plots get really show
        QtWidgets.QApplication.processEvents()
        

        self.start_button.clicked.connect(self.start_button_clicked)
        self.browse_button.clicked.connect(self.save_file_dialog)
        self.save_button.clicked.connect(self.save_data_clicked)
        self.load_settings_button.clicked.connect(self.load_settings)
        self.save_settings_button.clicked.connect(self.save_settings)
        self.lockin_spinner.valueChanged.connect(self.lockins_changed)
        

        self.updating_common_gui = True
        self.fs_combo.currentIndexChanged.connect(self.fs_changed)
        self.update_freq_combo.currentIndexChanged.connect(self.update_freq_changed)
        self.sub_blocks_combo.currentIndexChanged.connect(self.sub_blocks_changed)

        self.fs_combo.clear()
        self.fs_combo.insertItems(0,self.lockin_widgets[0].LCRsettings.fs_name_list)
        self.fs_combo.setCurrentIndex(self.lockin_widgets[0].LCRsettings.fs_setting)

        self.update_freq_combo.clear()
        self.update_freq_combo.insertItems(0,self.lockin_widgets[0].LCRsettings.update_freq_name_list)
        self.update_freq_combo.setCurrentIndex(self.lockin_widgets[0].LCRsettings.update_freq_setting)

        self.sub_blocks_combo.clear()
        self.sub_blocks_combo.insertItems(0,self.lockin_widgets[0].LCRsettings.sub_blocks_name_list)
        self.sub_blocks_combo.setCurrentIndex(self.lockin_widgets[0].LCRsettings.sub_blocks_setting)

        self.setWindowTitle('TiePieLCR '+self.lockin_widgets[0].LCRsettings.version)

        self.updating_common_gui = False

        self.multiprocessing_init()

        self.state = 0

        #time.sleep(4)
        self.sync_settings()

    def lockins_changed(self):
        new_lockins = self.lockin_spinner.value()
        if new_lockins < self.lockins:
            while new_lockins < self.lockins:
                self.tabWidget.removeTab(self.lockins-1)
                self.lockin_widgets.pop()
                self.lockins = self.lockins - 1
        else:
            while new_lockins > self.lockins:
                self.lockin_widgets.append(lockin_tab(self,self.lockins))
                self.tabWidget.insertTab(self.lockins,self.lockin_widgets[self.lockins], 'lockin'+str(self.lockins+1))
                self.lockins = self.lockins + 1

        self.sync_settings()


    def tcp_daemon(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while(1):
            try:
                sock.bind(('127.0.0.1', 12345))
                sock.listen(1)
                break
            except:
                self.displayed_error.value = b'could not bind. Try killing all python instances in taskmanager.'
                time.sleep(0.1)
                continue

        
        while(1):
            connection,address = sock.accept()  
            while True:  
                try:
                    buf = connection.recv(16000)
                except:
                    break
                if not buf: break
                #print(buf) 
                if buf == b'start':
                    if self.start_button.text() == 'start':
                        self.start_button.click()
                        connection.send(b'ack')
                    else:
                        self.start_button.click()
                        time.sleep(0.5)
                        self.start_button.click()
                        connection.send(b'ack')
                elif buf == b'stop':
                    if self.start_button.text() == 'stop':
                        self.start_button.click()
                        connection.send(b'ack')
                    else:
                        self.start_button.click()
                        time.sleep(0.5)
                        self.start_button.click()
                        connection.send(b'ack')
                elif buf[0:5] == b'save:':
                    filename_string = buf[5:].decode("utf-8")
                    self.save_data(filename_string)
                    connection.send(b'ack')
                elif buf[0:16] == b'settings_pickle:':
                    pickle_string = buf[16:]
                    settings = pickle.loads(pickle_string)
                    self.lockin_widgets[self.tabWidget.currentIndex()].LCRsettings = settings
                    self.lockin_widgets[self.tabWidget.currentIndex()].LCRsettings.load_calibration()
                    self.sync_settings()
                    self.lockin_widgets[self.tabWidget.currentIndex()].gui_update_required = True
                    while not self.settings_queue.empty():
                        time.sleep(0.03)
                    connection.send(b'ack')
                elif buf == b'settings_pickle?':
                    settings_string = pickle.dumps(self.lockin_widgets[self.tabWidget.currentIndex()].LCRsettings)
                    connection.send(b'settings_pickle: ' + settings_string)
                elif buf == b'impedance_pickles?':
                    if (len(self.lockin_widgets[0].dem_time_vec)<=1):
                        connection.send(b'nak')
                        continue
                    data = []
                    for i1 in range(self.lockins):
                        data.append(impedance())
                        data[i1].set_timestamp(self.lockin_widgets[i1].int_timestamp,self.lockin_widgets[i1].dem_time_vec[-1])
                        data[i1].set_values(self.lockin_widgets[i1].int_demodulate_1,self.lockin_widgets[i1].int_demodulate_2)
                        data[i1].set_errors(self.lockin_widgets[i1].int_demodulte_1_error,self.lockin_widgets[i1].int_demodulte_2_error)
                        data[i1].set_offsets(self.lockin_widgets[i1].int_ref_offset,self.lockin_widgets[i1].int_ref_offset)
                        data[i1].set_clipping(self.lockin_widgets[i1].ref_clipping_during_integration,self.lockin_widgets[i1].sig_clipping_during_integration)
                    data_string = pickle.dumps(data)                    
                    connection.send(b'impedance_pickles: ' + data_string)
                elif buf == b'impedance_pickle?':
                    data = impedance()
                    if (len(self.lockin_widgets[self.tabWidget.currentIndex()].dem_time_vec)<=1):
                        connection.send(b'nak')
                        continue
                    data.set_timestamp(self.lockin_widgets[self.tabWidget.currentIndex()].int_timestamp,self.lockin_widgets[self.tabWidget.currentIndex()].dem_time_vec[-1])
                    data.set_values(self.lockin_widgets[self.tabWidget.currentIndex()].int_demodulate_1,self.lockin_widgets[self.tabWidget.currentIndex()].int_demodulate_2)
                    data.set_errors(self.lockin_widgets[self.tabWidget.currentIndex()].int_demodulte_1_error,self.lockin_widgets[self.tabWidget.currentIndex()].int_demodulte_2_error)
                    data.set_offsets(self.lockin_widgets[self.tabWidget.currentIndex()].int_ref_offset,self.lockin_widgets[self.tabWidget.currentIndex()].int_ref_offset)
                    data.set_clipping(self.lockin_widgets[self.tabWidget.currentIndex()].ref_clipping_during_integration,self.lockin_widgets[self.tabWidget.currentIndex()].sig_clipping_during_integration)
                    data_string = pickle.dumps(data)                    
                    connection.send(b'impedance_pickle: ' + data_string)
                else:
                    connection.send(b'nack')

                        
        connection.close()

    def multiprocessing_init(self):
        #start a seperate thread for updating the plots.
        print("running multiprocessing init")
        self.shared_state = Value('i',0)
        self.stored_data_requested = Value('i',0)
        
        self.plot_data_queue = Queue()
        self.stored_data_queue = Queue()

        self.acquisition_process = Process(target=self.acquisition_function, args=(self.settings_queue, self.displayed_error, self.shared_state, self.plot_data_queue,self.stored_data_queue,self.stored_data_requested))
        #self.sync_settings()
        self.acquisition_process.start()

        print("started acquisition process")
        #self.thread = threading.Thread(target=self.plotting_daemon, args=(), daemon=True)
        #self.thread.start()

        self.tcp_thread = threading.Thread(target=self.tcp_daemon, args=(), daemon=True)
        self.tcp_thread.start()

        timer = QTimer(self)
        timer.timeout.connect(self.state_timer)
        timer.start(10)

    @staticmethod
    def acquisition_function(LCR_settings_queue,displayed_error,shared_state,plot_data_queue,stored_data_queue,stored_data_requested):
        acquisition_object = acquisition(LCR_settings_queue,displayed_error,shared_state,plot_data_queue,stored_data_queue,stored_data_requested)

    def state_timer(self):
        if self.state != 0 and self.shared_state.value == 0:
            self.state = 0
            self.start_button.setText('start')

        if self.state != 2 and self.shared_state.value == 2:
            self.state = 2
        
        if self.shared_state.value == 4:
            self.start_button.setText('start')
            self.shared_state.value = 0
            self.state = 0
            
        

        text = self.displayed_error.value
        self.statusbar.showMessage(text.decode("utf-8"),10)
        #self.error_label.setText(text.decode("utf-8") )

    def save_settings(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self,"","", options=options)
        if directory:
            for i1 in range(self.lockins):
                filename_string = directory+'/'+self.settings_filename+'_'+str(i1+1)+'.yaml'
                print('Saved settings to: ' + filename_string)
                self.lockin_widgets[i1].LCRsettings.save_settings(filename_string)
                self.displayed_error.value = b"settings saved in " + bytes(filename_string, 'utf-8')

    def load_settings(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self,"","", options=options)
        if directory:
            for i1 in range(self.lockins):
                filename_string = directory+'/'+self.settings_filename+'_'+str(i1+1)+'.yaml'
                print('loaded settings from: ' + filename_string)
                self.lockin_widgets[i1].LCRsettings.load_settings(filename_string,i1)
            self.sync_settings()
            self.lockin_widgets[self.tabWidget.currentIndex()].gui_update_required = True
            while not self.settings_queue.empty():
                time.sleep(0.03)

    def get_all_stored_data(self):
        package ={}
        self.stored_data_requested.value = 1
        package = self.stored_data_queue.get()
        return package

    def save_data_clicked(self):
        save_location =  self.save_line.text()
        if len(save_location) > 0:
            self.save_data(save_location)

    def start_button_clicked(self):
        if self.state == 0:
            self.start_measurement()
        elif self.state == 2:
            self.stop_measurement()
        else:
            print("start button pressed while bussy")
            print(str(self.state))        

    def save_data(self,save_location):
        impedance_data = {}
        impedance_time = []
        ref_offset = []
        sig_offset = []
        offset_time = []
        timestamps = []
        settings_dict = []
        package = self.get_all_stored_data()
        if 'empty' in package:
            self.displayed_error.value = b"error: There is no data to plot. Click on start to record some data."
            return False
        timestamps = np.zeros((len(package[0]['timestamps']),1))
        timestamps[:,0] = package[0]['timestamps']
        for i1 in range(self.lockins):
            
            settings_dict.append(self.lockin_widgets[i1].LCRsettings.get_settings_dict(self.settings_folder+self.settings_filename + '_' + str(i1+1)+'.yaml'))
            impedance_data['lockin'+str(i1+1)] = package[i1]['demodulation_data']
            impedance_time.append(package[i1]['demodulation_time'])
            ref_offset.append(package[i1]['ref_offset'])
            sig_offset.append(package[i1]['sig_offset'])
            offset_time.append(package[i1]['offset_time'])

        #sio.savemat(save_location, { 'impedance_time':impedance_time})
        #sio.savemat(save_location, {'impedance_data':impedance_data})
        sio.savemat(save_location, {'impedance_data':impedance_data, 'impedance_time':impedance_time,'ref_offset':ref_offset,'sig_offset':sig_offset, 'offset_time':offset_time, 'settings':settings_dict,'timestamps':timestamps })
        
        self.displayed_error.value = b"data saved in " + bytes(save_location, 'utf-8')

    def save_file_dialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()","","Matlab (*.mat)", options=options)
        if fileName:
            self.save_line.setText(fileName)
            self.save_data_clicked()

    def sync_settings(self):
        if self.state > -1:
            if self.updating_gui == False:
                all_settings = []
                for i1 in range(self.lockins):
                    all_settings.append(self.lockin_widgets[i1].LCRsettings)
                    if i1 > 0:
                        all_settings[i1].set_sub_blocks(self.lockin_widgets[0].LCRsettings.sub_blocks_setting)
                        all_settings[i1].set_sample_frequency(self.lockin_widgets[0].LCRsettings.fs_setting)
                        all_settings[i1].set_update_freq(self.lockin_widgets[0].LCRsettings.update_freq_setting)

                print("syncing settings")
                self.settings_queue.put(all_settings)
                time.sleep(0.001)
                for i1 in range(self.lockins):
                    all_settings[i1].reset()

    def fs_changed(self):
        if self.updating_common_gui == False:
            value = self.fs_combo.currentIndex()
            for i1 in range(self.lockins):
                package = self.lockin_widgets[i1].LCRsettings.set_sample_frequency(value)
                if 'error' in package:
                    self.displayed_error.value = package['error']
                    return False
            self.sync_settings()
            self.fft_sens_result_label.setText(str(self.lockin_widgets[0].LCRsettings.get_fft_sensitivity()))
    
    def sub_blocks_changed(self):
        if self.updating_common_gui == False:
            value = self.sub_blocks_combo.currentIndex()
            for i1 in range(self.lockins):
                package = self.lockin_widgets[i1].LCRsettings.set_sub_blocks(value)
                if 'error' in package:
                    self.displayed_error.value = package['error']
                    return False
            
            self.sync_settings()
            self.fft_sens_result_label.setText(str(self.lockin_widgets[0].LCRsettings.get_fft_sensitivity()))


    def update_freq_changed(self):
        if self.updating_common_gui == False:
            value = self.update_freq_combo.currentIndex()
            for i1 in range(self.lockins):
                package = self.lockin_widgets[i1].LCRsettings.set_update_freq(value)
                if 'error' in package:
                    self.displayed_error.value = package['error']
                else:
                    self.displayed_error.value = b''
            self.sync_settings()
            self.fft_sens_result_label.setText(str(self.lockin_widgets[0].LCRsettings.get_fft_sensitivity()))

    def start_measurement(self):
        self.start_button.setText('stop')
        self.displayed_error.value = b''

        self.shared_state.value = 1
        self.state = 1

    def stop_measurement(self):
        self.shared_state.value = 3
        self.state = 3
        self.start_button.setText('start')

    def closeEvent(self, event):
        self.shared_state.value = 3
        self.state = 3

        files_in_directory = os.listdir('./'+self.settings_folder)
        filtered_files = [file for file in files_in_directory if file.endswith(".yaml")]
        for file in filtered_files:
            path_to_file = os.path.join('./'+self.settings_folder, file)
            os.remove(path_to_file)

        for i1 in range(self.lockins):
            filename_string = self.settings_folder+self.settings_filename + '_' + str(i1+1)+'.yaml'
            self.lockin_widgets[i1].LCRsettings.save_settings(filename_string)
        time.sleep(0.001)
        self.acquisition_process.terminate()



   

def main():
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    #pg.setConfigOption('useOpenGL',1)
    #pg.setConfigOption('useCupy',1)

    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()

    sys.exit(app.exec_())    

if __name__ == '__main__':
    freeze_support()      
    main()