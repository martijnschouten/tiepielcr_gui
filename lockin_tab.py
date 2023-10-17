"""
.. module:: lockin_tab
   :synopsis: This class contains the code for the main GUI
.. moduleauthor:: Martijn Schouten <github.com/martijnschouten>
"""

from distutils.log import error
import numpy as np
from PyQt5 import QtWidgets, uic
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from TiePieLCR_settings import TiePieLCR_settings
import matplotlib as mpl
import math

class lockin_tab(QtWidgets.QWidget):
    sig_color = (255, 0, 0)
    pen_sig = pg.mkPen(color=(255, 0, 0))
    ref_color = (0, 0, 255)
    pen_ref = pg.mkPen(color=(0, 0, 255))
    
    host = "127.0.0.1"
    port = 65432
    buffer_size = 5
    
    color_list = ((230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (0, 0, 0))
    
    dem_time_vec = []
    
    int_demodulate_1 = 0
    int_demodulate_2 = 0
    gui_update_required = False

    first_time_reference_changed = True

    
    def __init__(self, mainwindow,instance,*args, **kwargs):
        super(lockin_tab, self).__init__(*args, **kwargs)
            
        #Load the UI Page
        uic.loadUi('lockin.ui', self)

        self.mainwindow = mainwindow
        self.instance = instance
        self.LCRsettings = TiePieLCR_settings()

        filestring = self.mainwindow.settings_folder+self.mainwindow.settings_filename + '_' + str(self.instance+1)+'.yaml'
        print('filestring:' + filestring)
        if not self.LCRsettings.load_settings(filestring,instance):
            self.mainwindow.displayed_error.value = b"could not load settings. using default settings"
        settings= self.LCRsettings
        if not settings.load_calibration():
            self.LCRsettings = settings
            self.mainwindow.displayed_error.value = b"could not load calibration. Using default calibration"
        
        #self.mainwindow.settings_queue.put(self.LCRsettings)

           
            

        #connect all the signals of the interface
        
        self.amplitude_spinner.valueChanged.connect(self.amplitude_changed)
        self.offset_spinner.valueChanged.connect(self.offset_changed)
        self.sig_range_combo.currentIndexChanged.connect(self.sig_range_changed)
        self.ref_range_combo.currentIndexChanged.connect(self.ref_range_changed)
        self.sig_coupling_combo.currentIndexChanged.connect(self.sig_coupling_changed)
        self.ref_coupling_combo.currentIndexChanged.connect(self.ref_coupling_changed)
        self.reference_combo.currentIndexChanged.connect(self.reference_changed)
        self.gain_combo.currentIndexChanged.connect(self.gain_changed)
        
        self.no_frontend_check.stateChanged.connect(self.no_frontend_changed)
        self.dem_is_gen_check.stateChanged.connect(self.dem_is_gen_changed)
        self.insert_gen_freq_button.pressed.connect(self.insert_gen_freq_pressed)
        self.delete_gen_freq_button.pressed.connect(self.delete_gen_freq_pressed)
        self.insert_dem_freq_button.pressed.connect(self.insert_dem_freq_pressed)
        self.delete_dem_freq_button.pressed.connect(self.delete_dem_freq_pressed)
        self.periods_spinner.valueChanged.connect(self.periods_changed)
        self.gen_frequency_table.cellChanged.connect(self.gen_frequency_table_changed)
        self.dem_frequency_table.cellChanged.connect(self.dem_frequency_table_changed)
        self.fmin_spinner.valueChanged.connect(self.fmin_changed)
        self.fmax_spinner.valueChanged.connect(self.fmax_changed)
        self.plot_time_spinner.valueChanged.connect(self.plot_time_changed)
        self.integration_spinner.valueChanged.connect(self.integration_changed)
        self.bandwidth_spinner.valueChanged.connect(self.bandwidth_changed)
        self.offset_bandwidth_spinner.valueChanged.connect(self.offset_bandwidth_changed)
        self.offset_integration_spinner.valueChanged.connect(self.offset_integration_changed)
        self.optimise_button.pressed.connect(self.optimise_pressed)
        self.frontend_spinner.valueChanged.connect(self.frontend_number_changed)

        self.format_combo.currentIndexChanged.connect(self.format_changed)
        
        #build all the plots.
        self.build_sig_fft_plot()
        self.build_ref_fft_plot()
        self.build_signal_plot()
        self.build_demodulate_plots()
        
        #update the components that can't be update through the tcp interface
        self.update_gui()

        #make it possible to update some components through the tcp interface
        timer = QTimer(self)
        timer.timeout.connect(self.dynamic_update_gui)
        timer.start(10)

        timer = QTimer(self)
        timer.timeout.connect(self.plotting_timer)
        timer.start(40)

        QtWidgets.QApplication.processEvents()
        
        self.reference = None
        self.signal = None
        self.time_vec = None
        self.reference_fft = None
        self.signal_fft = None
        self.freq_vec = None
        self.dem_time_vec = None
        self.dem_1_vec = None
        self.dem_2_vec = None
        self.int_timestamp = None
        self.int_demodulate_1 = None
        self.int_demodulate_2 = None
        self.ref_clipping_during_integration = None
        self.sig_clipping_during_integration = None
        self.int_demodulte_1_error = None
        self.int_demodulte_2_error = None
        self.int_ref_offset = None
        self.int_sig_offset = None
        

    def update_gui(self):
        print("update gui")

        self.mainwindow.updating_gui = True

        if not self.LCRsettings.load_calibration():
            self.mainwindow.displayed_error.value = b"could not load calibration. Using default calibration"
        

        self.reference_combo.clear()
        self.reference_combo.insertItems(0,self.LCRsettings.calibration.reference_name_list)
        self.reference_combo.setCurrentIndex(self.LCRsettings.get_reference_setting())
        print(self.reference_combo.currentIndex())

        #clear all components from the list, then add all elements and set the currently selected item
        #these values come from the settings file loaded into LCRsettings.
        self.gain_combo.clear()
        self.gain_combo.insertItems(0,self.LCRsettings.calibration.gain_name_list)
        self.gain_combo.setCurrentIndex(self.LCRsettings.get_gain_setting())

        self.ref_range_combo.clear()
        self.ref_range_combo.insertItems(0,self.LCRsettings.scope_range_name_list)
        self.ref_range_combo.setCurrentIndex(self.LCRsettings.get_reference_scope_range_index())

        self.sig_range_combo.clear()
        self.sig_range_combo.insertItems(0,self.LCRsettings.scope_range_name_list)
        self.sig_range_combo.setCurrentIndex(self.LCRsettings.get_signal_scope_range_index())
        self.set_y_range()

        self.ref_coupling_combo.clear()
        self.ref_coupling_combo.insertItems(0,self.LCRsettings.scope_coupling_name_list)
        self.ref_coupling_combo.setCurrentIndex(self.LCRsettings.get_reference_scope_coupling())


        self.sig_coupling_combo.clear()
        self.sig_coupling_combo.insertItems(0,self.LCRsettings.scope_coupling_name_list)
        self.sig_coupling_combo.setCurrentIndex(self.LCRsettings.get_signal_scope_coupling())

        self.format_combo.clear()
        self.format_combo.insertItems(0,self.LCRsettings.impedance_formats)
        self.format_combo.setCurrentIndex(self.LCRsettings.impedance_format)

        #set the values of all the spinners.
        self.amplitude_spinner.setValue(self.LCRsettings.get_gen_amplitude())
        self.offset_spinner.setValue(self.LCRsettings.get_gen_offset())
        self.periods_spinner.setValue(self.LCRsettings.get_plot_periods())
        self.fmin_spinner.setValue(self.LCRsettings.get_minimum_plot_frequency())
        self.fmax_spinner.setValue(self.LCRsettings.get_maximum_plot_frequency())
        self.bandwidth_spinner.setValue(self.LCRsettings.get_demodulation_bandwidth())
        self.plot_time_spinner.setValue(self.LCRsettings.get_plot_time())
        self.plot_time_spinner.setValue(self.LCRsettings.get_integration_time())
        self.offset_integration_spinner.setValue(self.LCRsettings.get_offset_integration_time())
        self.offset_bandwidth_spinner.setValue(self.LCRsettings.get_offset_bandwidth())
        self.frontend_spinner.setValue(self.LCRsettings.get_frontend_number())
        self.integration_spinner.setValue(self.LCRsettings.get_integration_time())

        self.mainwindow.fft_sens_result_label.setText(str(self.LCRsettings.get_fft_sensitivity()))
        self.fundamental_result_label.setText(str(self.LCRsettings.get_f_fun()))
        self.crest_factor_result_label.setText("%0.3f"%(self.LCRsettings.get_multisine_crest_factor()))

        print(self.LCRsettings.get_multisine_freqs())
        freqs = self.LCRsettings.get_multisine_freqs()
        weights = self.LCRsettings.get_multisine_weights()
        self.gen_frequency_table.setRowCount(len(freqs))
        for i1 in range(len(freqs)):
            self.gen_frequency_table.setItem(i1,0,QtWidgets.QTableWidgetItem(str(freqs[i1])))
            self.gen_frequency_table.setItem(i1,1,QtWidgets.QTableWidgetItem(str(weights[i1])))

        print(self.LCRsettings.get_demodulation_freqs())
        freqs = self.LCRsettings.get_demodulation_freqs()
        tiepies = self.LCRsettings.get_demodulation_tiepies()
        self.dem_frequency_table.setRowCount(len(freqs))
        for i1 in range(len(freqs)):
            item0 = QtWidgets.QTableWidgetItem(str(freqs[i1]))
            item0.setForeground(QtGui.QBrush(QtGui.QColor(self.color_list[i1][0],self.color_list[i1][1],self.color_list[i1][2])))
            item1 = QtWidgets.QTableWidgetItem(str(tiepies[i1]))
            item1.setForeground(QtGui.QBrush(QtGui.QColor(self.color_list[i1][0],self.color_list[i1][1],self.color_list[i1][2])))
            self.dem_frequency_table.setItem(i1,0,item0)
            self.dem_frequency_table.setItem(i1,1,item1)

        self.dem_is_gen_check.setCheckState(self.LCRsettings.dem_is_gen)
        self.dem_is_gen_check.setTristate(False)

        self.no_frontend_check.setCheckState(self.LCRsettings.no_frontend)
        self.no_frontend_check.setTristate(False)

        self.mainwindow.updating_gui = False

        self.set_multisine()
        

        if self.dem_is_gen_check.checkState():
            self.insert_dem_freq_button.setEnabled(False)
            self.delete_dem_freq_button.setEnabled(False)
            self.dem_frequency_table.setEnabled(False)
            self.sync_tables()
        else:
            self.insert_dem_freq_button.setEnabled(True)
            self.delete_dem_freq_button.setEnabled(True)
            self.dem_frequency_table.setEnabled(True)

    #some components need to be reloaded when they are updated through the tcp interface
    #this cannot be done from the tcp daemon since gui components can only be updated from
    #the same threat.
    def dynamic_update_gui(self):     
        if self.gui_update_required == True:
            self.update_gui()
            self.gui_update_required = False
            
        

    

    

    def build_signal_plot(self):
        #if not self.mainwindow.plot_data_queue is None:
        #    self.mainwindow.plot_data_queue.queue.clear()
        self.sig_graph = self.lck1_sig_graph.getPlotItem()

        #make this a yyplot
        self.ref_graph = pg.ViewBox()
        self.sig_graph.showAxis('right')
        self.sig_graph.scene().addItem(self.ref_graph)
        self.sig_graph.getAxis('right').linkToView(self.ref_graph)
        self.ref_graph.setXLink(self.sig_graph)

        #define a curve for each graph to which the data can be added later on
        self.sig_curve = self.sig_graph.plot(pen=self.pen_sig)
        self.ref_curve = pg.PlotCurveItem(pen=self.pen_ref)
        self.ref_graph.addItem(self.ref_curve)

        self.format_sig_label()
        self.format_ref_label()

        xaxis = self.sig_graph.getAxis('bottom')
        xaxis.setLabel("Time (s)")

        self.update_signal_plot_views()
        self.sig_graph.vb.sigResized.connect(self.update_signal_plot_views)

    def update_signal_plot_views(self):
        self.ref_graph.setGeometry(self.sig_graph.vb.sceneBoundingRect())
        self.ref_graph.linkedViewChanged(self.sig_graph.vb, self.ref_graph.XAxis)

    def build_sig_fft_plot(self):
        self.sig_fft_graph = self.lck1_sig_fft_graph.getPlotItem()
            
        self.sig_fft_graph.disableAutoRange()
        self.sig_fft_graph.setLogMode(True, True)
        self.sig_fft_curve = self.sig_fft_graph.plot(pen=self.pen_sig)
        self.format_sig_fft_label()
        xaxis = self.sig_fft_graph.getAxis('bottom')
        xaxis.setLogMode(True)
        xaxis.enableAutoSIPrefix(False)
        xaxis.setLabel("Frequency", "Hz")

    def build_ref_fft_plot(self):
        self.ref_fft_graph = self.lck1_ref_fft_graph.getPlotItem()
        
        self.ref_fft_graph.disableAutoRange()
        self.ref_fft_graph.setLogMode(True, True)
        self.ref_fft_curve = self.ref_fft_graph.plot(pen=self.pen_ref)
        self.format_ref_fft_label()
        xaxis = self.ref_fft_graph.getAxis('bottom')
        xaxis.setLogMode(True)
        xaxis.enableAutoSIPrefix(False)
        xaxis.setLabel("Frequency","Hz")
            

    def build_demodulate_plots(self):
        
        self.mainwindow.updating_gui = True

        self.demodulate_1_graph = self.lck1_demodulate_graph_1.getPlotItem()
        self.demodulate_1_curve = [None for _ in range(self.LCRsettings.get_number_of_demodulate_freqs())] 
        self.demodulate_1_graph.clear()

        self.demodulate_2_graph = self.lck1_demodulate_graph_2.getPlotItem()
        self.demodulate_2_curve = [None for _ in range(self.LCRsettings.get_number_of_demodulate_freqs())]
        self.demodulate_2_graph.clear()

        
        for i2 in range(self.LCRsettings.get_number_of_demodulate_freqs()):
            self.demodulate_1_curve[i2] = self.demodulate_1_graph.plot(pen=pg.mkPen(color=self.color_list[i2]))
            self.demodulate_2_curve[i2] = self.demodulate_2_graph.plot(pen=pg.mkPen(color=self.color_list[i2]))
        xaxis = self.demodulate_1_graph.getAxis('bottom')
        xaxis.setLabel("Time (s)")

        xaxis = self.demodulate_2_graph.getAxis('bottom')
        xaxis.setLabel("Time (s)")
        self.format_demodulate_labels()
        self.mainwindow.updating_gui = False
            
    def format_demodulate_labels(self):
        yaxis1 = self.demodulate_1_graph.getAxis('left')
        yaxis1.enableAutoSIPrefix(True)
        yaxis1.setWidth(w=70)
        yaxis2 = self.demodulate_2_graph.getAxis('left')
        yaxis2.enableAutoSIPrefix(True)
        yaxis2.setWidth(w=70)
        yaxis1.setLabel(self.LCRsettings.get_impedance_format_label1(),self.LCRsettings.get_impedance_format_unit1())
        yaxis2.setLabel(self.LCRsettings.get_impedance_format_label2(),self.LCRsettings.get_impedance_format_unit2())

    def format_sig_label(self):  
        #make the adc plots look nice  
        yaxisleft = self.sig_graph.getAxis('left')
        yaxisleft.enableAutoSIPrefix(True)
        yaxisleft.setWidth(w=60)
        rgbColor = 'rgb(' + str(self.sig_color[0]) + ',' + str(self.sig_color[1]) + ',' + str(self.sig_color[2]) + ')' 
        Style = {'color': rgbColor}
        yaxisleft.setLabel('Signal','V',**Style)

    def format_ref_label(self):
        yaxisright = self.sig_graph.getAxis('right')
        yaxisright.enableAutoSIPrefix(True)
        yaxisright.setWidth(w=60)
        rgbColor = 'rgb(' + str(self.ref_color[0]) + ',' + str(self.ref_color[1]) + ',' + str(self.ref_color[2]) + ')'
        Style = {'color': rgbColor}
        unit = self.LCRsettings.get_reference_unit()
        yaxisright.setLabel('Reference',unit,**Style)

    def format_sig_fft_label(self): 
        #make the adc plots look nice  
        yaxisleft = self.sig_fft_graph.getAxis('left')
        yaxisleft.enableAutoSIPrefix(False)
        yaxisleft.setLogMode(True)
        yaxisleft.setWidth(w=70)
        rgbColor = 'rgb(' + str(self.sig_color[0]) + ',' + str(self.sig_color[1]) + ',' + str(self.sig_color[2]) + ')' 
        Style = {'color': rgbColor}
        yaxisleft.setLabel('Signal (V/√Hz)',**Style)

    def format_ref_fft_label(self):
        yaxisleft = self.ref_fft_graph.getAxis('left')
        yaxisleft.enableAutoSIPrefix(False)
        yaxisleft.setLogMode(True)
        yaxisleft.setWidth(w=70)
        rgbColor = 'rgb(' + str(self.ref_color[0]) + ',' + str(self.ref_color[1]) + ',' + str(self.ref_color[2]) + ')'
        Style = {'color': rgbColor}
        unit = self.LCRsettings.get_reference_unit()
        yaxisleft.setLabel('Reference (' + unit + '/√Hz)',**Style)     

    def set_y_range(self):
        #if self.mainwindow.state > -1:
        if self.LCRsettings.get_gain_setting() == 0:
            self.sig_fft_graph.setYRange(-8,1)
        else:
            self.sig_fft_graph.setYRange(-9,-1)

        sig_range = self.LCRsettings.get_signal_scope_range_value()/self.LCRsettings.get_gain_value()
        self.sig_graph.setYRange(-sig_range,sig_range)

        setting = self.LCRsettings.get_reference_setting()
        if  setting == 0:
            self.ref_fft_graph.setYRange(-14,-5)
        elif setting == 1:
            self.ref_fft_graph.setYRange(-11,-3)
        elif setting == 2:
            self.ref_fft_graph.setYRange(-12,-4)
        elif setting == 3:
            self.ref_fft_graph.setYRange(-17,-9)
        elif setting == 4:
            self.ref_fft_graph.setYRange(-18,-10)
        elif setting == 5 or setting == 6:
            self.ref_fft_graph.setYRange(-9,-1)  
        else:
            print("unkown reference selected")

        ref_range = self.LCRsettings.get_reference_scope_range_value()*self.LCRsettings.get_reference_gain()
        self.ref_graph.setYRange(-ref_range,ref_range)

        



    

    def format_changed(self):
        if self.mainwindow.updating_gui == False:
            value = self.format_combo.currentIndex()
            self.LCRsettings.set_impedance_format(value)
            self.mainwindow.sync_settings()
            self.build_demodulate_plots()

    def periods_changed(self):
        value = self.periods_spinner.value()
        self.LCRsettings.set_plot_periods(value)
        self.mainwindow.sync_settings()

    def fmin_changed(self):
        value = self.fmin_spinner.value()
        self.LCRsettings.set_minimum_plot_frequency(value)
        self.mainwindow.sync_settings()

    def fmax_changed(self):
        value = self.fmax_spinner.value()
        self.LCRsettings.set_maximum_plot_frequency(value)
        self.mainwindow.sync_settings()

    
    def gen_frequency_table_changed(self):
        if self.mainwindow.updating_gui == False:
            self.set_multisine()
            if self.dem_is_gen_check.checkState():
                self.sync_tables()
            self.set_demodulation_freqs()
            self.mainwindow.sync_settings()

    def dem_frequency_table_changed(self):
        if self.mainwindow.updating_gui == False:
            self.set_demodulation_freqs()
            self.build_demodulate_plots()
            self.mainwindow.sync_settings()

    def set_demodulation_freqs(self):
        print("set_demodulation_freqs")
        rows = self.dem_frequency_table.rowCount()
        freq_list = []
        tiepie_list = []
        for i1 in range(rows):
            item0 = self.dem_frequency_table.item(i1,0)
            item1 = self.dem_frequency_table.item(i1,1)
            if item0 is None or item1 is None:
                return False
            try:
                freq_list.append(float(item0.text()))
            except:
                freq_list.append(12345)
                #self.mainwindow.displayed_error.value = b'error: non nummeric frequency entered'
            try:
                tiepie_list.append(int(float(item1.text())))
            except:
                tiepie_list.append(1)
                #self.mainwindow.displayed_error.value = b'error: non nummeric tiepie entered'
            
        package = self.LCRsettings.set_demodulation_params(freq_list,tiepie_list)
        if 'error' in package:
            self.mainwindow.displayed_error.value = package['error']

    def update_colors(self):
        old_updating_gui_state = self.mainwindow.updating_gui
        self.mainwindow.updating_gui = True
        rows = self.dem_frequency_table.rowCount()
        for i1 in range(rows):
            item0 = self.dem_frequency_table.item(i1,0)
            item0_new = QtWidgets.QTableWidgetItem(item0)
            item0_new.setForeground(QtGui.QBrush(QtGui.QColor(self.color_list[i1][0],self.color_list[i1][1],self.color_list[i1][2])))
            self.dem_frequency_table.setItem(i1,0,item0_new)
            
            item1 = self.dem_frequency_table.item(i1,1)
            item1_new = QtWidgets.QTableWidgetItem(item1)
            item1_new.setForeground(QtGui.QBrush(QtGui.QColor(self.color_list[i1][0],self.color_list[i1][1],self.color_list[i1][2])))
            self.dem_frequency_table.setItem(i1,1,item1_new)
        self.mainwindow.updating_gui = old_updating_gui_state

    def sync_tables(self):
        if self.mainwindow.updating_gui == False:
            print("sync_tables")
            self.mainwindow.updating_gui = True
            rows = self.gen_frequency_table.rowCount()
            self.dem_frequency_table.setRowCount(rows)
            for i1 in range(rows):
                item0 = self.gen_frequency_table.item(i1,0)
                item0_new = QtWidgets.QTableWidgetItem(item0)
                self.dem_frequency_table.setItem(i1,0,item0_new)
                item1_new = QtWidgets.QTableWidgetItem(str(self.instance+1))
                self.dem_frequency_table.setItem(i1,1,item1_new)
            self.update_colors()
            self.set_demodulation_freqs()
            self.build_demodulate_plots()
            self.mainwindow.updating_gui = False
            

    def set_multisine(self):
        if self.mainwindow.updating_gui == False:
            print("set_multisine")
            rows = self.gen_frequency_table.rowCount()
            freq_list = []
            weight_list = []
            for i1 in range(rows):
                item0 = self.gen_frequency_table.item(i1,0)
                item1 = self.gen_frequency_table.item(i1,1)
                if item0 is None or item1 is None:
                    return False
                freq_list.append(float(item0.text()))
                weight_list.append(float(item1.text()))
            package = self.LCRsettings.set_multisine(freq_list,weight_list)
            if not 'error' in package:
                self.fundamental_result_label.setText(str(self.LCRsettings.get_f_fun()))
                self.crest_factor_result_label.setText("%0.3f"%(self.LCRsettings.get_multisine_crest_factor()))
            else:
                self.mainwindow.displayed_error.value = package['error']
                self.fundamental_result_label.setText('invalid')
                self.crest_factor_result_label.setText('invalid')

    def amplitude_changed(self):
        if self.mainwindow.updating_gui == False:
            text = b"Amplitude + offset exceeds maximum voltage!"
            if not self.LCRsettings.set_amplitude(self.amplitude_spinner.value()):
                self.mainwindow.displayed_error.value = text
            else:
                if self.mainwindow.displayed_error.value == text:
                    self.mainwindow.displayed_error.value = b''
                self.mainwindow.sync_settings()

    def offset_changed(self):
        if self.mainwindow.updating_gui == False:
            text = b"Amplitude + offset exceeds maximum voltage!"
            if not self.LCRsettings.set_offset(self.offset_spinner.value()):
                self.mainwindow.displayed_error.value = text
            else:
                if self.mainwindow.displayed_error.value == text:
                    self.mainwindow.displayed_error.value = b''
                self.mainwindow.sync_settings()
    def ref_range_changed(self):
        if self.mainwindow.updating_gui == False:
            self.LCRsettings.set_reference_scope_range(self.ref_range_combo.currentIndex())
            self.set_y_range()
            self.mainwindow.sync_settings()

    def sig_range_changed(self):
        if self.mainwindow.updating_gui == False:
            self.LCRsettings.set_signal_scope_range(self.sig_range_combo.currentIndex())
            self.set_y_range()
            self.mainwindow.sync_settings()

    def ref_coupling_changed(self):
        if self.mainwindow.updating_gui == False:
            self.LCRsettings.set_reference_scope_coupling(self.ref_coupling_combo.currentIndex())
            self.mainwindow.sync_settings()

    def sig_coupling_changed(self):
        if self.mainwindow.updating_gui == False:
            self.LCRsettings.set_signal_scope_coupling(self.sig_coupling_combo.currentIndex())
            self.mainwindow.sync_settings()

    def reference_changed(self):
        if self.mainwindow.updating_gui == False:
            print('reference changed:' + str(self.reference_combo.currentIndex()))
            self.LCRsettings.set_reference(self.reference_combo.currentIndex())
            self.format_ref_label()
            self.set_y_range()
            self.mainwindow.sync_settings()

    def gain_changed(self):
        print("gain change in %s"%(self.instance))
        if self.mainwindow.updating_gui == False:
            self.LCRsettings.set_LCR_gain(self.gain_combo.currentIndex())
            self.format_sig_label()
            self.set_y_range()
            self.mainwindow.sync_settings()

    def bandwidth_changed(self):
        if self.mainwindow.updating_gui == False:
            package =self.LCRsettings.set_demodulation_bandwidth(self.bandwidth_spinner.value())
            if 'error' in package:
                self.mainwindow.displayed_error.value = package['error']
            else:
                self.mainwindow.sync_settings()

    def offset_bandwidth_changed(self):
        if self.mainwindow.updating_gui == False:
            package = self.LCRsettings.set_offset_bandwidth(self.offset_bandwidth_spinner.value())
            self.mainwindow.sync_settings()

    def offset_integration_changed(self):
        if self.mainwindow.updating_gui == False:
            package = self.LCRsettings.set_offset_integration_time(self.offset_integration_spinner.value())
            self.mainwindow.sync_settings()        

    def plot_time_changed(self):
        if self.mainwindow.updating_gui == False:
            self.LCRsettings.set_plot_time(self.plot_time_spinner.value())
            self.mainwindow.sync_settings()

    def integration_changed(self):
        if self.mainwindow.updating_gui == False:
            self.LCRsettings.set_integration_time(self.integration_spinner.value())
            self.mainwindow.sync_settings()

    def optimise_pressed(self):
        self.LCRsettings.optimise_crest()
        self.mainwindow.sync_settings()
        self.crest_factor_result_label.setText("%0.3f"%(self.LCRsettings.get_multisine_crest_factor()))
        

    def insert_gen_freq_pressed(self):
        self.gen_frequency_table.insertRow(self.gen_frequency_table.currentRow()+1)
        if self.dem_is_gen_check.checkState():
            self.dem_frequency_table.insertRow(self.gen_frequency_table.currentRow()+1)

    def delete_gen_freq_pressed(self):
        self.gen_frequency_table.removeRow(self.gen_frequency_table.currentRow())
        if self.dem_is_gen_check.checkState():
            self.dem_frequency_table.removeRow(self.gen_frequency_table.currentRow())
        self.gen_frequency_table_changed()

    def insert_dem_freq_pressed(self):
        self.dem_frequency_table.insertRow(self.dem_frequency_table.currentRow()+1)
        self.update_colors()

    def delete_dem_freq_pressed(self):
        self.dem_frequency_table.removeRow(self.dem_frequency_table.currentRow())
        self.dem_frequency_table_changed()
        self.update_colors()

    def dem_is_gen_changed(self):
        if self.mainwindow.updating_gui == False:
            if self.dem_is_gen_check.checkState():
                self.insert_dem_freq_button.setEnabled(False)
                self.delete_dem_freq_button.setEnabled(False)
                self.dem_frequency_table.setEnabled(False)
                self.sync_tables()
            else:
                self.insert_dem_freq_button.setEnabled(True)
                self.delete_dem_freq_button.setEnabled(True)
                self.dem_frequency_table.setEnabled(True)

            self.LCRsettings.dem_is_gen = self.dem_is_gen_check.checkState()
            self.mainwindow.sync_settings()    

    def no_frontend_changed(self):
        if self.mainwindow.updating_gui == False:
            if self.no_frontend_check.checkState():
                self.reference_combo.setEnabled(False)
                self.gain_combo.setEnabled(False)
            else:
                self.reference_combo.setEnabled(True)
                self.gain_combo.setEnabled(True)

            self.LCRsettings.no_frontend = bool(self.no_frontend_check.checkState())
            self.mainwindow.sync_settings()       

    def frontend_number_changed(self):
        if self.mainwindow.updating_gui == False:
            self.LCRsettings.set_frontend_number(self.frontend_spinner.value())
            self.update_gui()
            self.mainwindow.sync_settings()

    def plotting_timer(self):
        
        if self.instance == self.mainwindow.tabWidget.currentIndex():
            new_data = False
            if self.mainwindow.plot_data_queue:
                while not self.mainwindow.plot_data_queue.empty():
                    try:
                        package_list = self.mainwindow.plot_data_queue.get_nowait()
                        new_data = True
                    except:
                        new_data = False



            if new_data == True:
                #print("len package list" + str(len(package_list)))
                for i1 in range(self.mainwindow.lockins):
                    package = package_list[i1]
                    self.mainwindow.lockin_widgets[i1].reference = package['plot_reference']
                    self.mainwindow.lockin_widgets[i1].signal = package['plot_signal']
                    self.mainwindow.lockin_widgets[i1].time_vec = package['plot_time']
                    self.mainwindow.lockin_widgets[i1].reference_fft = package['plot_reference_fft']
                    self.mainwindow.lockin_widgets[i1].signal_fft = package['plot_signal_fft']
                    self.mainwindow.lockin_widgets[i1].freq_vec = package['plot_frequency']
                    self.mainwindow.lockin_widgets[i1].dem_time_vec = package['plot_demodulate_time']
                    self.mainwindow.lockin_widgets[i1].dem_1_vec = package['plot_demodulate_1']
                    self.mainwindow.lockin_widgets[i1].dem_2_vec = package['plot_demodulate_2']
                    self.mainwindow.lockin_widgets[i1].int_timestamp = package['int_timestamp']
                    self.mainwindow.lockin_widgets[i1].int_demodulate_1 = package['int_demodulate_1']
                    self.mainwindow.lockin_widgets[i1].int_demodulate_2 = package['int_demodulate_2']
                    self.mainwindow.lockin_widgets[i1].ref_clipping_during_integration = package['ref_clipping_during_integration']
                    self.mainwindow.lockin_widgets[i1].sig_clipping_during_integration = package['sig_clipping_during_integration']
                    self.mainwindow.lockin_widgets[i1].int_demodulte_1_error = package['int_demodulate_1_error']
                    self.mainwindow.lockin_widgets[i1].int_demodulte_2_error = package['int_demodulate_2_error']
                    self.mainwindow.lockin_widgets[i1].int_ref_offset = package['int_ref_offset']
                    self.mainwindow.lockin_widgets[i1].int_sig_offset = package['int_sig_offset']

                self.sig_graph.disableAutoRange(pg.ViewBox.XAxis)
                self.ref_graph.disableAutoRange(pg.ViewBox.XAxis)
                self.sig_curve.setData(self.time_vec, self.signal)
                self.ref_curve.setData(self.time_vec, self.reference)
                self.sig_graph.enableAutoRange(pg.ViewBox.XAxis)
                self.ref_graph.enableAutoRange(pg.ViewBox.XAxis)


                signal_fft_abs = np.absolute(self.signal_fft)
                reference_fft_abs = np.absolute(self.reference_fft)
                self.sig_fft_curve.setData(self.freq_vec, signal_fft_abs)
                self.ref_fft_curve.setData(self.freq_vec, reference_fft_abs)

                value1 = ''
                value2 = ''
                label1 = ''
                label2 = ''
                formatter = mpl.ticker.EngFormatter(places=4)
                for i1 in range(self.LCRsettings.get_number_of_demodulate_freqs()):
                    if not self.demodulate_1_curve[i1] is None:
                        self.demodulate_1_graph.disableAutoRange()
                        self.demodulate_1_curve[i1].setData(self.dem_time_vec,self.dem_1_vec[:,i1])
                        self.demodulate_1_graph.enableAutoRange()
                    if not self.demodulate_2_curve[i1] is None:
                        self.demodulate_2_graph.disableAutoRange()
                        self.demodulate_2_curve[i1].setData(self.dem_time_vec,self.dem_2_vec[:,i1])
                        self.demodulate_2_graph.enableAutoRange()
                    if math.isinf(self.int_demodulate_1[i1]):
                        value1 = value1 + 'Inf' + self.LCRsettings.get_impedance_format_unit1() + '\n'
                    elif math.isnan(self.int_demodulate_1[i1]):
                        value1 = value1 + 'NaN' + self.LCRsettings.get_impedance_format_unit1() + '\n'
                    else:
                        value1 = value1 + formatter(self.int_demodulate_1[i1]) + self.LCRsettings.get_impedance_format_unit1() + '\n'
                    if math.isinf(self.int_demodulate_2[i1]):
                        value2 = value2 + 'Inf' + self.LCRsettings.get_impedance_format_unit2() + '\n'
                    elif math.isnan(self.int_demodulate_1[i1]):
                        value2 = value2 + 'NaN' + self.LCRsettings.get_impedance_format_unit2() + '\n'    
                    else:
                        value2 = value2 + formatter(self.int_demodulate_2[i1]) + self.LCRsettings.get_impedance_format_unit2() + '\n'

                    label1 = label1 + "<span style=\"color:#%x%x%x;\" > %s%s :</span><br>"%(self.color_list[i1][0],self.color_list[i1][1],self.color_list[i1][2],self.LCRsettings.get_impedance_format_label1(),str(self.LCRsettings.dem_freqs[i1]))
                    label2 = label2 + "<span style=\"color:#%x%x%x;\" > %s%s :</span><br>"%(self.color_list[i1][0],self.color_list[i1][1],self.color_list[i1][2],self.LCRsettings.get_impedance_format_label2(),str(self.LCRsettings.dem_freqs[i1]))
                
                self.value1_value.setText(value1)
                self.value2_value.setText(value2)
                self.value1_label.setText(label1)
                self.value2_label.setText(label2)

                if math.isinf(self.int_ref_offset):
                    self.ref_offset_value.setText('Inf')
                elif math.isnan(self.int_ref_offset):
                    self.ref_offset_value.setText('NaN')
                else:
                    self.ref_offset_value.setText(formatter(self.int_ref_offset)+self.LCRsettings.get_reference_offset_unit())
                
                result = math.isinf(self.int_sig_offset)
                if result:
                    self.sig_offset_value.setText('Inf')
                elif math.isnan(self.int_sig_offset):
                    self.ref_offset_value.setText('NaN')
                else:
                    self.sig_offset_value.setText(formatter(self.int_sig_offset)+'V')

                if self.ref_clipping_during_integration or self.sig_clipping_during_integration:
                    self.value1_value.setStyleSheet('color: rgb(170, 0, 0)')
                    self.value2_value.setStyleSheet('color: rgb(170, 0, 0)')
                else:
                    self.value1_value.setStyleSheet('color: rgb(0, 85, 0);')
                    self.value2_value.setStyleSheet('color: rgb(0, 85, 0);')

                if self.instance == 0:
                    serial_numbers = package['serial_numbers']
                    for i1 in range(len(serial_numbers)):
                        if i1 < len(self.mainwindow.lockin_widgets):
                            self.mainwindow.lockin_widgets[i1].serial_value.setText(str(serial_numbers[i1]))

                new_data = False 