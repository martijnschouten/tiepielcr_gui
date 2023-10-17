import yaml
import io
import numpy as np

class TiePieLCR_calibration:
    def __init__(self):
        self.reference_gain_list = [1,-1.0/2.61e3,-1.0/200e3,-390e-12,-3.9e-12,1.0/2.61e3,1]
        self.reference_unit_list = ['', 'A','A','C','C','A' , 'V']
        self.reference_offset_unit_list = ['', 'A','A','A','A','A' , 'V']
        self.reference_name_list = ['None','Lcur 370 μA/V', 'Lcur 5 μA/V', 'Lcur 390 pC/V', 'Lcur 3.9 pC/V', 'HcurI', 'HcurV']

        self.gain_name_list = ['1x', '50x']
        self.gain_list = [1,50]

        #these should be calibrated
        self.reference_R = [1,2.61e3,200e3,1.0e8,1.0e10,2.7e3,1]
        self.reference_C = [0,3.9e-12,3.9e-12,390e-12,3.9e-12,3.9e-12,0]
        self.protection_R = [0,0,0,2.61e3,0,0,0]
        self.protection_C = [0,0,0,3.9e-12,0,0,0]

        self.z_error_mag_rel = [[[0],[0],[0],[0],[0],[0],[0]],[[0],[0],[0],[0],[0],[0],[0]]]
        self.z_C_offset = [[[0],[0],[0],[0],[0],[0],[0]],[[0],[0],[0],[0],[0],[0],[0]]]
        self.z_error_angle = [[[0],[0],[0],[0],[0],[0],[0]],[[0],[0],[0],[0],[0],[0],[0]]]

        self.calibration_dict = {}

    def load_calibration(self, filename):
        
        try:
            with open(filename, 'r', encoding='utf-8') as stream:
                calibration = yaml.safe_load(stream)
                print("Succefully loaded calibration from: "+filename)
        except Exception as e:
            print(e)
            print("Loading calibration failed.")
            return False

        self.calibration_dict = calibration
        
        if 'reference_R' in calibration:
             self.reference_R = np.double(np.array(calibration['reference_R']))
        if 'reference_C' in calibration:
             self.reference_C = np.double(np.array(calibration['reference_C']))
        if 'protection_R' in calibration:
             self.protection_R = np.double(np.array(calibration['protection_R']))
        if 'protection_C' in calibration:
             self.protection_C = np.double(np.array(calibration['protection_C']))

        if 'reference_gain_list' in calibration:
            self.reference_gain_list = calibration['reference_gain_list']
        if 'reference_unit_list' in calibration:
            self.reference_unit_list = calibration['reference_unit_list']
        if 'reference_offset_unit_list' in calibration:
            self.reference_offset_unit_list = calibration['reference_offset_unit_list']
        if 'reference_name_list' in calibration:
            self.reference_name_list = calibration['reference_name_list']

        if 'gain_name_list' in calibration:
            self.gain_name_list = calibration['gain_name_list']
        if  'gain_list' in calibration:
            self.gain_list = calibration['gain_list']
        
        if 'z_error_mag_rel' in calibration:
            for i1 in range(len(calibration['z_error_mag_rel'])):
                for i2 in range(len(self.z_error_mag_rel[i1])):
                    self.z_error_mag_rel[i1][i2] = np.double(np.array(calibration['z_error_mag_rel'][i1][i2]))
        if 'C_offset' in calibration:
            for i1 in range(len(calibration['C_offset'])):
                for i2 in range(len(self.z_C_offset[i1])):
                    self.z_C_offset[i1][i2] = np.double(np.array(calibration['C_offset'][i1][i2]))
        
        if 'z_error_angle' in calibration:
            for i1 in range(len(calibration['z_error_angle'])):
                for i2 in range(len(self.z_error_mag_rel[i1])):
                    self.z_error_angle[i1][i2] = np.double(np.array(calibration['z_error_angle'][i1][i2]))
        return True
    
    def get_calibration_dict(self,filename):
        self.save_calibration(filename)
        self.load_calibration(filename)
        return self.calibration_dict

    def set_poly_calibration(self,f,reference,gain,Zmeas,Ztest,deg):
        if Zmeas.ndim == 1:
            magn = np.absolute(Zmeas)
            angle = np.angle(Zmeas)
            magn_test = np.absolute(Ztest)
            angle_test = np.angle(Ztest)

            error_mag_rel = (magn-magn_test)/magn
            error_angle = angle-angle_test

            f_log = np.log10(f)
            self.z_error_mag_rel[gain][reference] = np.polyfit(f_log, error_mag_rel, deg)
            self.z_error_angle[gain][reference] = np.polyfit(f_log, error_angle, deg)
            self.z_C_offset[gain][reference] = np.zeros(deg)
        else:
            if len(Zmeas[:,0]) == 1:
                magn = np.absolute(Zmeas[0,:])
                angle = np.angle(Zmeas[0,:])
                magn_test = np.absolute(Ztest[0,:])
                angle_test = np.angle(Ztest[0,:])

                error_mag_rel = (magn-magn_test)/magn
                error_angle = angle-angle_test

                f_log = np.log10(f)
                self.z_error_mag_rel[gain][reference] = np.polyfit(f_log, error_mag_rel, deg)
                self.z_error_angle[gain][reference] = np.polyfit(f_log, error_angle, deg)
                self.z_C_offset[gain][reference] = np.zeros(deg)

            elif len(Zmeas[:,0]) == 2:
                #magn = np.absolute(Zmeas)
                #angle = np.angle(Zmeas)
                magn_test = np.absolute(Ztest)
                angle_test = np.angle(Ztest)

                a = np.real(Zmeas)
                b = np.imag(Zmeas)
                Rp_meas = (b**2+a**2)/a
                Cp_meas = -b/Rp_meas/a/2/np.pi/f

                a = np.real(Ztest)
                b = np.imag(Ztest)
                Rp_test = (b**2+a**2)/a
                Cp_test = -b/Rp_test/a/2/np.pi/f

                C_offset = Cp_meas[0,:]-Cp_test[0,:]
                
                
                Cp_meas = Cp_meas-C_offset
                Zmeas_comp = Rp_meas/(1+2*np.pi*f*1j*Cp_meas*Rp_meas)
                magn = np.absolute(Zmeas_comp)
                angle = np.angle(Zmeas_comp)

                error_mag_rel = (magn[1,:]-magn[0,:]-(magn_test[1,:]-magn_test[0,:]))/(magn[1,:]-magn[0,:])
                error_angle = np.mean(angle-angle_test,axis=0)
                

                f_log = np.log10(f)
                self.z_C_offset[gain][reference] = np.polyfit(f_log, C_offset, deg)
                self.z_error_mag_rel[gain][reference] = np.polyfit(f_log, error_mag_rel, deg)
                self.z_error_angle[gain][reference] = np.polyfit(f_log, error_angle, deg)
            else:
                raise Exception('can handle max 2 impedances per frequency at the moment')

            

        

    def get_recalibration_factor(self,old_cal,freqs,reference_setting,gain_setting):
        recalibration_factor = np.zeros(len(freqs),dtype=np.complex128)
        Z_old = old_cal.get_measurement_z(freqs,reference_setting,gain_setting)
        Z_new = self.get_measurement_z(freqs,reference_setting,gain_setting)
        for i1 in len(freqs):
            recalibration_factor[i1] = Z_new[i1]/Z_old[i1]
        return recalibration_factor

    def get_compensated_z(self,freqs,reference_setting,gain_setting,Z):
        p_offset = np.poly1d(self.z_C_offset[gain_setting][reference_setting])


        a = np.real(Z)
        b = np.imag(Z)
        Rp_meas = (b**2+a**2)/a
        Cp_meas = -b/Rp_meas/a/2/np.pi/freqs
        Cp_comp = Cp_meas-p_offset(np.log10(freqs))
        Z_comp = Rp_meas/(1+2*np.pi*freqs*1j*Cp_comp*Rp_meas)

        return Z_comp

    def get_measurement_z(self,freqs,reference_setting,gain_setting):
        """Calculate the impedance used in the feedback of the currently selected transimpedance amplifier for a specific frequency, taking into account stability caps, bias resistors and anti-aliassing filters.

        :param freqs: The frequency at which the impedance should be calculated
        :typ freqs: float
        :return: The impedance
        :rtype: Complex double
        """
        R_real = self.reference_R[reference_setting]
        C_real = self.reference_C[reference_setting]
        R_prot = self.protection_R[reference_setting]
        C_prot = self.protection_C[reference_setting]
        if R_prot==0 and C_prot==0:
            Z = R_real/(1j*R_real*freqs*2*np.pi*C_real+1)
        else:
            Z = R_real/(1j*R_real*freqs*2*np.pi*C_real+1)+R_prot/(1j*R_prot*freqs*2*np.pi*C_prot+1)

        freqs = np.array(freqs)
        freqs[freqs<10] = 10
        Z_mag = np.absolute(Z)
        Z_angle = np.angle(Z)
        f_log = np.log10(freqs)

        p_mag = np.poly1d(self.z_error_mag_rel[gain_setting][reference_setting])
        p_angle = np.poly1d(self.z_error_angle[gain_setting][reference_setting])
        error_mag_rel = p_mag(f_log)
        error_angle = p_angle(f_log)
        Z_mag_comp = Z_mag*(1-error_mag_rel)
        Z_angle_comp = Z_angle-error_angle
        Z = Z_mag_comp*np.exp(Z_angle_comp*1j)
        return Z

    def save_calibration(self,filename):

        calibration = {}
        calibration['reference_R'] = np.array(self.reference_R).tolist()
        calibration['reference_C'] = np.array(self.reference_C).tolist()
        calibration['protection_R'] = np.array(self.protection_R).tolist()
        calibration['protection_C'] = np.array(self.protection_C).tolist()

        calibration['reference_gain_list'] = self.reference_gain_list
        calibration['reference_unit_list'] = self.reference_unit_list
        calibration['reference_offset_unit_list'] = self.reference_offset_unit_list
        calibration['reference_name_list'] = self.reference_name_list

        calibration['gain_name_list'] = self.gain_name_list
        calibration['gain_list'] = self.gain_list


        calibration['z_error_mag_rel'] = self.z_error_mag_rel
        for i1 in range(len(self.z_error_mag_rel)):
            for i2 in range(len(self.z_error_mag_rel[i1])):
                calibration['z_error_mag_rel'][i1][i2] = np.array(self.z_error_mag_rel[i1][i2]).tolist()

        calibration['C_offset'] = self.z_C_offset
        for i1 in range(len(self.z_C_offset)):
            for i2 in range(len(self.z_C_offset[i1])):
                calibration['C_offset'][i1][i2] = np.array(self.z_C_offset[i1][i2]).tolist()

        calibration['z_error_angle'] = self.z_error_angle
        for i1 in range(len(self.z_error_angle)):
            for i2 in range(len(self.z_error_mag_rel[i1])):
                calibration['z_error_angle'][i1][i2] = np.array(self.z_error_angle[i1][i2]).tolist()

        with io.open(filename, 'w', encoding='utf8') as outfile:
                yaml.dump(calibration, outfile, default_flow_style=False, allow_unicode=True)